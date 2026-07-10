-- =====================================================================
-- 004_align_ml_feature_store.sql — spójny kontrakt cech XGBoost
-- Uruchom w Supabase SQL Editor PO 003_add_embedding.sql.
--
-- Fast Loop predykuje na 7 cechach. Ta migracja dopisuje brakujące
-- kolumny wejściowe do prediction_logs i odbudowuje ml_feature_store tak,
-- by Slow Loop trenował model na identycznych nazwach cech.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS av_relevance_avg FLOAT,
    ADD COLUMN IF NOT EXISTS news_volume_24h INTEGER,
    ADD COLUMN IF NOT EXISTS high_relevance_count INTEGER,
    ADD COLUMN IF NOT EXISTS av_llm_agreement FLOAT;

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
        actual_price_after_12h AS target_price
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
    target_price
FROM base;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);

CREATE OR REPLACE FUNCTION refresh_ml_feature_store()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY ml_feature_store;
END;
$$;
