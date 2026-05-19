-- migrations/007_council_votes.sql
-- Strukturalny audit trail rady doradczej — jedna linia per inwestor.
--
-- Cel: umożliwić odpytywanie "jak Burry głosował na NVDA w ostatnim miesiącu"
-- bez parsowania JSONB `prediction_logs.council_verdict`. Łatwiejszy backtest
-- spójności person, ranking inwestorów po confidence vs trafność, agregacje
-- per recommendation w czasie.
--
-- Relacja: prediction_id → prediction_logs.id (ON DELETE CASCADE — jeśli
-- usuniemy predykcję, znikają jej głosy).
--
-- Idempotentne (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS council_votes (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   UUID NOT NULL REFERENCES prediction_logs(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    investor_name   VARCHAR(64) NOT NULL,
    recommendation  VARCHAR(8)  NOT NULL CHECK (recommendation IN ('BUY','SELL','HOLD')),
    confidence      FLOAT       NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    reasoning       TEXT,
    key_factors     JSONB
);

CREATE INDEX IF NOT EXISTS idx_council_votes_symbol_timestamp
    ON council_votes (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_council_votes_investor_timestamp
    ON council_votes (investor_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_council_votes_prediction
    ON council_votes (prediction_id);

COMMENT ON TABLE council_votes IS
    'Per-investor advisory council votes. One row per investor per prediction.';
