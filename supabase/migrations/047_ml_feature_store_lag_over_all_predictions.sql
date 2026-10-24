-- =====================================================================
-- 047_ml_feature_store_lag_over_all_predictions.sql — `price_delta` w widoku
-- ml_feature_store liczony po WSZYSTKICH zalogowanych predykcjach (ML-17).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: inference (`get_last_prediction_price`) bierze cenę ostatniej
-- ZALOGOWANEJ predykcji symbolu — rozliczonej czy nie. Widok do 039 liczył LAG
-- w CTE z filtrem `actual_price_after_12h IS NOT NULL`, czyli po samych
-- rozliczonych wierszach. Zgodność train/serve trzymał więc tylko niezapisany
-- niezmiennik „predykcje rozliczają się w kolejności i przed następną". Wiersz
-- rozliczony po kolejnej predykcji albo nigdy nierozliczony (porzucony po
-- `reflection_max_age_hours`) dawał w treningu deltę od starszej ceny niż
-- w inference. Przykład: ceny 100 (rozliczona), 110 (otwarta), 121 (rozliczona)
-- — inference widziała 121/110 − 1 = 0,10, widok liczył 121/100 − 1 = 0,21.
--
-- ZMIANA: LAG po całej `prediction_logs`; filtr rozliczenia przechodzi do
-- zewnętrznego SELECT, obok filtra flag jakości (026/039), który już był PO
-- LAG. Reszta widoku odtworzona 1:1 z 039: kolumny, COALESCE-y (w tym
-- `alpha_fusion_score` → 0.0), `feature_set_version` bez COALESCE, target.
-- DROP MATERIALIZED VIEW kasuje indeksy — oba odtwarzamy (unikalny po `id`
-- jest wymagany do REFRESH ... CONCURRENTLY w `refresh_ml_feature_store`).
--
-- Nazwy `price_prev_12h` / `price_prev_24h` zostają (kontrakt odczytu), ale
-- ich prawdziwą semantykę opisuje COMMENT ON COLUMN poniżej.
--
-- Uprawnienia jak po 038/039: jawny REVOKE dla `anon` / `authenticated`
-- (z guardem pg_roles — gołego kontenera testowego te role nie mają);
-- `service_role` czyta z domyślnych uprawnień Supabase.
--
-- Idempotentne: DROP ... IF EXISTS + CREATE + CREATE INDEX IF NOT EXISTS,
-- REVOKE i COMMENT można powtarzać.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml_feature_store;

CREATE MATERIALIZED VIEW ml_feature_store AS
WITH base AS (
    SELECT
        id,
        symbol,
        timestamp,
        price_at_prediction AS price_current,
        -- LAG po WSZYSTKICH zalogowanych predykcjach symbolu (ML-17), jak
        -- referencja inference `get_last_prediction_price`.
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
-- Tylko rozliczone predykcje mają etykietę (ML-17: filtr PO LAG). Atrapa LLM
-- i każdy niepełny feed AV to nie dane treningowe (ML-12, EXP-07, DAT-07, DAT-08).
WHERE actual_price_after_12h IS NOT NULL
  AND NOT (
    data_quality_flags ?| ARRAY[
        'llm_analysis_failed',
        'av_keys_exhausted',
        'av_unexpected_payload',
        'av_fetch_error'
    ]
);

-- Indeksy odtwarzane (DROP MATERIALIZED VIEW usuwa je razem z widokiem).
-- Unikalny indeks po `id` jest wymagany do REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);

-- Role API Supabase bez uprawnień do odtworzonego widoku (BEZ-05, jak 038).
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format('REVOKE ALL ON ml_feature_store FROM %I', api_role);
        END IF;
    END LOOP;
END
$migration$;

COMMENT ON MATERIALIZED VIEW ml_feature_store IS
    'Cechy treningowe XGBoost: rozliczone predykcje bez flag llm_analysis_failed, av_keys_exhausted (026), av_unexpected_payload, av_fetch_error (039). LAG po wszystkich zalogowanych predykcjach, przed filtrami (047). feature_set_version z 027.';

COMMENT ON COLUMN ml_feature_store.price_prev_12h IS
    'Cena POPRZEDNIEJ zalogowanej predykcji symbolu (LAG 1 po całej prediction_logs, także nierozliczonych i oflagowanych) — ta sama referencja co get_last_prediction_price w inference. Nazwa historyczna: odstęp to odstęp kolejnych predykcji, nie 12 h.';

COMMENT ON COLUMN ml_feature_store.price_prev_24h IS
    'Cena predykcji sprzed DWÓCH zalogowanych predykcji symbolu (LAG 2 po całej prediction_logs). Nazwa historyczna: odstęp to dwie kolejne predykcje, nie 24 h. Nie jest cechą modelu.';
