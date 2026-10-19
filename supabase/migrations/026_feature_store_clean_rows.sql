-- =====================================================================
-- 026_feature_store_clean_rows.sql — trening bez wierszy-atrap (ML-12, EXP-07)
-- Uruchom PO 025_implied_edge.sql (ręczny workflow „DB Migrate").
--
-- PO CO: `data_quality_flags` (migracja 008) zapowiadała filtr przy treningu,
-- ale nikt flag nie czytał — do `ml_feature_store` trafiały wiersze, które nie
-- są realną predykcją ani realnymi danymi:
--   * `llm_analysis_failed` — neutralna atrapa „SIDEWAYS 50%" po awarii LLM
--     (ML-12): `llm_trend_signal` = 0 i `av_llm_agreement` = 0.5 to fallback,
--     nie sygnał;
--   * `av_keys_exhausted` — rotacja wyczerpała klucze Alpha Vantage, więc
--     cechy newsowe/sentymentu to zera „brak danych", nie „spokojny dzień"
--     (EXP-07).
-- `price_delta_no_prior_prediction` ZOSTAJE: to pierwsza predykcja symbolu,
-- jej `price_delta` i tak jest NULL w widoku (LAG bez poprzednika), a reszta
-- cech jest prawdziwa. Pomiar produkcji (2026-10-18/19, 288 rozliczonych
-- wierszy): `av_keys_exhausted` 45, `price_delta_no_prior_prediction` 38,
-- `llm_analysis_failed` 21 — filtr zostawia ok. 225 wierszy.
--
-- KOREKTA KRYTYCZNA (train/serve-skew): filtr jest WYŁĄCZNIE w zewnętrznym
-- SELECT, a `data_quality_flags` przechodzi przez CTE `base`. LAG liczy się więc
-- na pełnej sekwencji rozliczonych predykcji — tak jak inference
-- (`get_last_prediction_price` bierze cenę ostatniej zalogowanej predykcji bez
-- patrzenia na flagi). Filtr w CTE przesunąłby `price_prev_12h` wiersza po
-- atrapie na starszą predykcję, której inference nigdy nie widzi.
--
-- NULL-safe: `NULL ?| ARRAY[...]` daje NULL, a `WHERE NOT NULL` po cichu
-- wyciąłby wiersz, więc flagi wchodzą jako COALESCE(..., '[]').
--
-- Definicja odtworzona 1:1 z migracji 022 (8 cech, target = ZWROT 12h); jedyna
-- zmiana to filtr flag. DROP MATERIALIZED VIEW kasuje indeksy — oba odtwarzamy
-- (unikalny po `id` jest wymagany do REFRESH ... CONCURRENTLY).
--
-- Idempotentne: DROP ... IF EXISTS + CREATE + CREATE INDEX IF NOT EXISTS.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml_feature_store;

CREATE MATERIALIZED VIEW ml_feature_store AS
WITH base AS (
    SELECT
        id,
        symbol,
        timestamp,
        price_at_prediction AS price_current,
        LAG(price_at_prediction, 1) OVER (
            PARTITION BY symbol ORDER BY timestamp
        ) AS price_prev_12h,
        LAG(price_at_prediction, 2) OVER (
            PARTITION BY symbol ORDER BY timestamp
        ) AS price_prev_24h,
        sentiment_score,
        COALESCE(av_relevance_avg, 0.0) AS av_relevance_avg,
        COALESCE(news_volume_24h, 0) AS news_volume_24h,
        COALESCE(high_relevance_count, 0) AS high_relevance_count,
        CASE
            WHEN predicted_trend = 'BULLISH'  THEN  1
            WHEN predicted_trend = 'BEARISH'  THEN -1
            ELSE 0
        END AS llm_trend_signal,
        COALESCE(av_llm_agreement, 0.5) AS av_llm_agreement,
        -- Ósma cecha: NULL → 0.0, żeby dropna nie wyciął historii (skew).
        COALESCE(alpha_fusion_score, 0.0) AS alpha_fusion_score,
        -- Flagi jakości tylko do filtra w zewnętrznym SELECT (po LAG).
        COALESCE(data_quality_flags, '[]'::jsonb) AS data_quality_flags,
        actual_price_after_12h
    FROM prediction_logs
    WHERE actual_price_after_12h IS NOT NULL
)
SELECT
    id,
    symbol,
    timestamp,
    price_current,
    price_prev_12h,
    price_prev_24h,
    -- price_delta = zmiana względem POPRZEDNIEJ predykcji (guard anty-dziel-0).
    CASE
        WHEN price_prev_12h IS NULL OR price_prev_12h = 0 THEN NULL
        ELSE (price_current - price_prev_12h) / price_prev_12h
    END AS price_delta,
    sentiment_score,
    sentiment_score AS av_sentiment_score,
    av_relevance_avg,
    news_volume_24h,
    high_relevance_count,
    llm_trend_signal,
    av_llm_agreement,
    -- Ósma cecha ML: composite fuzji alfa (już z COALESCE 0.0 z base).
    alpha_fusion_score,
    -- Target = ZWROT 12h (guard anty-dziel-0 dla skażonych cen bazowych).
    CASE
        WHEN price_current IS NULL OR price_current = 0 THEN NULL
        ELSE (actual_price_after_12h - price_current) / price_current
    END AS target_return
FROM base
-- Atrapa LLM i wyczerpane klucze AV to nie dane treningowe (ML-12, EXP-07).
WHERE NOT (data_quality_flags ?| ARRAY['llm_analysis_failed', 'av_keys_exhausted']);

-- Indeksy odtwarzane (DROP MATERIALIZED VIEW usuwa je razem z widokiem).
-- Unikalny indeks po `id` jest wymagany do REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);

COMMENT ON MATERIALIZED VIEW ml_feature_store IS
    'Cechy treningowe XGBoost; bez wierszy z flagą llm_analysis_failed lub av_keys_exhausted (026). LAG liczony przed filtrem.';
