-- =====================================================================
-- 001_init.sql — pierwsza migracja StockAgent
-- Uruchom w Supabase SQL Editor (Project → SQL Editor → New query).
-- Idempotentne (IF NOT EXISTS / OR REPLACE).
-- =====================================================================

-- ---------------------------------------------------------------------
-- Rozszerzenia
-- ---------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- Tabela: prediction_logs
--   Audytowalna historia decyzji agenta. Każdy cykl 12h dopisuje rekord.
--   Pola feedback (actual_price_after_12h, accuracy_score, correction_insights)
--   uzupełnia węzeł `reflect` w kolejnych iteracjach.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_logs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol                   VARCHAR(20) NOT NULL,
    timestamp                TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Dane wejściowe — kontekst decyzji
    price_at_prediction      DECIMAL(14, 4),
    sentiment_score          FLOAT,
    news_summary             TEXT,

    -- Wyjście agenta
    predicted_trend          VARCHAR(20),   -- 'BULLISH' | 'BEARISH' | 'SIDEWAYS'
    predicted_target_price   DECIMAL(14, 4),
    reasoning_text           TEXT,

    -- Feedback loop (uzupełniane w kolejnych cyklach)
    actual_price_after_12h   DECIMAL(14, 4),
    accuracy_score           FLOAT,
    correction_insights      TEXT
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_symbol_timestamp
    ON prediction_logs (symbol, timestamp DESC);

-- Szybkie wyszukiwanie nieocenionych predykcji (Self-Reflection)
CREATE INDEX IF NOT EXISTS idx_prediction_logs_unverified
    ON prediction_logs (symbol, timestamp DESC)
    WHERE actual_price_after_12h IS NULL;

-- ---------------------------------------------------------------------
-- Widok zmaterializowany: ml_feature_store
--   Feature Store dla Slow Loop (trening XGBoost).
--   Agreguje historię + label-encoding sygnału LLM.
--   Refresh: REFRESH MATERIALIZED VIEW ml_feature_store; (przed treningiem)
-- ---------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS ml_feature_store AS
SELECT
    id,
    symbol,
    timestamp,

    -- Cechy rynkowe (Market Features)
    price_at_prediction                                                              AS price_current,
    LAG(price_at_prediction, 1) OVER (PARTITION BY symbol ORDER BY timestamp)        AS price_prev_12h,
    LAG(price_at_prediction, 2) OVER (PARTITION BY symbol ORDER BY timestamp)        AS price_prev_24h,

    -- Cechy społeczne
    sentiment_score,

    -- Cechy hybrydowe z LLM (label encoding)
    CASE
        WHEN predicted_trend = 'BULLISH'  THEN  1
        WHEN predicted_trend = 'BEARISH'  THEN -1
        ELSE 0
    END AS llm_trend_signal,

    -- Target (zmienna objaśniana — Y)
    actual_price_after_12h AS target_price
FROM
    prediction_logs
WHERE
    actual_price_after_12h IS NOT NULL;   -- bierzemy tylko zamknięte pętle

CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);             -- wymagane do CONCURRENTLY refresh

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);

-- ---------------------------------------------------------------------
-- (Opcjonalna) funkcja RPC do odświeżania widoku z poziomu klienta.
--   Wywołanie: supabase.rpc("refresh_ml_feature_store").execute()
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION refresh_ml_feature_store()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY ml_feature_store;
END;
$$;
