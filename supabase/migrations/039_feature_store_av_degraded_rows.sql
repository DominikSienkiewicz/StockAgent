-- =====================================================================
-- 039_feature_store_av_degraded_rows.sql — ml_feature_store bez wierszy
-- z niepełnym feedem Alpha Vantage (DAT-07, DAT-08).
-- Uruchom PO 038_enable_rls.sql (ręczny workflow „DB Migrate").
--
-- PO CO: klient AV oznacza dwa kolejne przypadki „brak danych, a nie spokojny
-- dzień" i graf zapisuje je w `prediction_logs.data_quality_flags`:
--   * `av_unexpected_payload` — NEWS_SENTIMENT zwróciło payload bez `feed`
--     i bez markera limitu (`Error Message`, `{}`) dla tego symbolu;
--   * `av_fetch_error` — błąd sieci/HTTP przerwał pobieranie feedu, więc
--     symbol dostał feed częściowy albo pusty.
-- Cechy newsów i sentymentu takiego wiersza to zera „bez danych", tak jak przy
-- `av_keys_exhausted` (026), więc nie trenują modelu.
--
-- Widok odtworzony 1:1 z migracji 027 (kolumny, COALESCE-y, `feature_set_version`
-- bez COALESCE), zmienia się tylko lista flag w filtrze. Filtr zostaje PO LAG:
-- `price_prev_12h` liczy się po wszystkich rozliczonych predykcjach, jak
-- referencja inference (`get_last_prediction_price`), która flag nie czyta.
-- DROP MATERIALIZED VIEW kasuje indeksy — oba odtwarzamy (unikalny po `id`
-- jest wymagany do REFRESH ... CONCURRENTLY w `refresh_ml_feature_store`).
--
-- Uprawnienia jak po 038: ALTER DEFAULT PRIVILEGES z 038 nie daje nowemu
-- widokowi grantów `anon` / `authenticated`; jawny REVOKE (z guardem pg_roles,
-- bo gołego kontenera testowego te role nie mają) domyka to także na bazie,
-- na której domyślne uprawnienia ktoś przywrócił ręcznie. `service_role`
-- dostaje odczyt z domyślnych uprawnień Supabase, jak przy 026/027.
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
-- Atrapa LLM i każdy niepełny feed AV to nie dane treningowe (ML-12, EXP-07,
-- DAT-07, DAT-08).
WHERE NOT (
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
    'Cechy treningowe XGBoost; bez wierszy z flagą llm_analysis_failed, av_keys_exhausted (026), av_unexpected_payload lub av_fetch_error (039). LAG liczony przed filtrem. feature_set_version z 027.';

COMMENT ON COLUMN prediction_logs.feature_set_version IS
    'Wersja zestawu cech (DAT-09): 1 = feed AV bez okna czasu, 2 = okno 24 h, 3 = feed leniwy per symbol (FIN-04); NULL = nieznana.';
