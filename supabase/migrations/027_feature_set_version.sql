-- =====================================================================
-- 027_feature_set_version.sql — wersja zestawu cech na predykcji (DAT-09)
-- Uruchom PO 026_feature_store_clean_rows.sql (ręczny workflow „DB Migrate").
--
-- PO CO: commit 3738ef2 ograniczył feed NEWS_SENTIMENT Alpha Vantage do
-- ostatnich 24 h (`time_from` = now − 24 h, `sort=LATEST`, filtr po
-- `time_published` w kliencie). Wcześniej `news_volume_24h` liczyło trafne
-- artykuły wśród 50 NAJNOWSZYCH w dowolnym wieku — kilka godzin dla NVDA,
-- miesiące dla rzadko opisywanego ETF-u. Nazwa cechy została (kontrakt ML),
-- więc model wspólny mieszałby dwa znaczenia tej samej kolumny. Ta migracja
-- dodaje `prediction_logs.feature_set_version` i wystawia ją w
-- `ml_feature_store`, żeby trening mógł czytać tylko bieżącą wersję
-- (`FEATURE_SET_VERSION` w `src/application/ml_features.py`).
--   1 — feed bez ograniczenia czasu (wiersze sprzed 3738ef2);
--   2 — feed ograniczony do 24 h.
--
-- REGUŁA BACKFILLU (z historii gita):
--   * 3738ef2 „fix(news): bound the Alpha Vantage feed to the last 24 hours"
--     ma datę autora i commitu 2026-10-18 23:47:22 +02:00 = 21:47:22 UTC.
--     Żaden wiersz sprzed tej chwili nie mógł powstać z oknem 24 h, więc
--     `timestamp < 2026-10-18 21:47:22 UTC` → wersja 1 (pewna).
--   * Produkcja (GitHub Actions) chodzi z `origin/main`. W chwili pisania tej
--     migracji `origin/main` = 961f65b (push 2026-10-18 22:20 +02:00) i NIE
--     zawiera 3738ef2; cron Fast Loopa jest wstrzymany od 80fa83d (2026-08-17),
--     więc nowe wiersze powstają tylko z ręcznego dispatchu. Wiersz z
--     późniejszym znacznikiem mógł przyjść ze starego kodu (wersja 1) albo już
--     z nowego (wersja 2) — moment wdrożenia nie jest zapisany w bazie.
--     Takie wiersze ZOSTAJĄ NULL („wersja nieznana"): filtr treningu po
--     bieżącej wersji ich nie bierze, więc znaczenia cechy się nie mieszają,
--     a migracja nie twierdzi niczego, czego nie wie. Kosztem jest kilka
--     wierszy mniej w próbce, nie skażony model.
--   * Po migracji wersję stempluje Fast Loop przy zapisie predykcji (flaga
--     `feature_set_version_enabled`); bez flagi nowe wiersze też zostają NULL.
--
-- Widok odtworzony 1:1 z migracji 026 (filtr flag jakości PO LAG) i rozszerzony
-- o jedną kolumnę. Kolumna wchodzi BEZ COALESCE: NULL = wersja nieznana, a nie
-- „pierwsza" — inaczej wiersze z okresu przejściowego udawałyby wersję 1.
-- DROP MATERIALIZED VIEW kasuje indeksy — oba odtwarzamy (unikalny po `id`
-- jest wymagany do REFRESH ... CONCURRENTLY).
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, backfill tylko po NULL-ach sprzed
-- daty odcięcia, DROP ... IF EXISTS + CREATE + CREATE INDEX IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Wersja zestawu cech, z którą policzono wejścia predykcji (DAT-09);
    -- NULL = nieznana (wiersz z okresu przejściowego albo zapis bez flagi).
    ADD COLUMN IF NOT EXISTS feature_set_version INTEGER;

UPDATE prediction_logs
SET feature_set_version = 1
WHERE feature_set_version IS NULL
  AND timestamp < TIMESTAMPTZ '2026-10-18 21:47:22+00';

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
        -- Wersja zestawu cech; NULL zostaje NULL (wersja nieznana).
        feature_set_version,
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
    -- Filtr treningu po bieżącej wersji robi `get_feature_store_data` (flaga).
    feature_set_version,
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
    'Cechy treningowe XGBoost; bez wierszy z flagą llm_analysis_failed lub av_keys_exhausted (026). LAG liczony przed filtrem. feature_set_version z 027.';

COMMENT ON COLUMN prediction_logs.feature_set_version IS
    'Wersja zestawu cech (DAT-09): 1 = feed AV bez okna czasu, 2 = okno 24 h; NULL = nieznana.';
