-- migrations/010_quota_alerts.sql
-- Tabela quota_alerts — persistent audit dla wyczerpań subskrypcji / limitów.
--
-- Cel: żaden 429 ani exhaustion klucza API nie ma prawa się zgubić.
-- Adaptery (AlphaVantage, OpenAI, Finnhub, Resend) emitują `QuotaAlert`
-- przez `QuotaMonitor`, main_agent persystuje po cyklu i przy następnym
-- mailu wyciąga history z ostatnich 24h do bannera.
--
-- Idempotentne (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS quota_alerts (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,                              -- "Alpha Vantage" / "OpenAI" / ...
    severity     TEXT NOT NULL CHECK (severity IN ('WARNING', 'CRITICAL')),
    message      TEXT NOT NULL,
    action       TEXT NOT NULL,
    occurred_at  TIMESTAMPTZ NOT NULL,
    inserted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indeks na occurred_at — pobieranie ostatnich N godzin do bannera.
CREATE INDEX IF NOT EXISTS idx_quota_alerts_occurred_at
    ON quota_alerts (occurred_at DESC);

COMMENT ON TABLE quota_alerts IS
    'Audit trail wyczerpań limitów / subskrypcji wykrytych przez adaptery.';
COMMENT ON COLUMN quota_alerts.source IS
    'Human-readable nazwa serwisu (Alpha Vantage / OpenAI / Finnhub / Resend).';
COMMENT ON COLUMN quota_alerts.severity IS
    'WARNING = degraded but functional. CRITICAL = data lost / operation failed.';
COMMENT ON COLUMN quota_alerts.action IS
    'Konkretny krok do podjęcia przez właściciela (np. "Add new key in .env").';
