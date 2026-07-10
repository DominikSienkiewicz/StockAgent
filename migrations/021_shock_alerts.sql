-- =====================================================================
-- 021_shock_alerts.sql — dedup alertów szoku rynkowego poza cyklem (#11)
-- Uruchom w Supabase SQL Editor PO 020_decision_receipts.sql.
--
-- Alert szoku poza cyklem: darmowy watch (Finnhub/CoinGecko) wykrywa gwałtowny
-- ruch ceny między cyklami digestu i pcha natychmiastowy push ("🚨 NVDA -6.2%").
-- Ta tabela to trwały rejestr wysłanych alertów, służący DEDUPLIKACJI.
--
-- WYMAGANIE WIĄŻĄCE: UNIQUE(symbol, alert_date) — twardy debounce "jeden alert
-- per symbol per dzień". Bez tego spam przy chaotycznym rynku (wiele szoków
-- tego samego dnia) uczy użytkownika ignorować kanał. Watch przed wysłaniem
-- próbuje wstawić wiersz; kolizja UNIQUE = alert już dziś poszedł, pomijamy.
--
-- Idempotentne (IF NOT EXISTS).
-- =====================================================================

CREATE TABLE IF NOT EXISTS shock_alerts (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT NOT NULL,
    -- Dzień alertu (DATE, nie timestamp) — granulacja debounce'u to jeden dzień.
    alert_date  DATE NOT NULL,
    -- Zmiana ceny, która wyzwoliła alert (np. -0.062 = -6.2%).
    delta       DOUBLE PRECISION NOT NULL,
    -- Kierunek szoku ('UP' / 'DOWN') — czytelny dla renderu i filtrów.
    direction   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Twardy debounce: jeden alert per symbol per dzień.
    CONSTRAINT uq_shock_alerts_symbol_date UNIQUE (symbol, alert_date)
);

-- Odczyt "czy dziś już alertowaliśmy dla symbolu" + historia ostatnich dni.
CREATE INDEX IF NOT EXISTS idx_shock_alerts_symbol_date
    ON shock_alerts (symbol, alert_date DESC);

COMMENT ON TABLE shock_alerts IS
    'Rejestr alertów szoku poza cyklem z twardym debounce UNIQUE(symbol, alert_date).';
COMMENT ON COLUMN shock_alerts.alert_date IS
    'Dzień alertu — granulacja debounce "jeden alert per symbol per dzień".';
