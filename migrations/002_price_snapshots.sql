-- =====================================================================
-- 002_price_snapshots.sql — tabela snapshotów cen
-- Uruchom w Supabase SQL Editor PO 001_init.sql.
--
-- Rozwiązuje cold-start deadlock: agent zapisuje bieżącą cenę w KAŻDYM
-- cyklu (niezależnie czy robi predykcję), dzięki czemu następny cykl ma
-- punkt odniesienia do policzenia delty.
--
-- `prediction_logs` pozostaje czyste — wyłącznie realne predykcje.
-- =====================================================================

CREATE TABLE IF NOT EXISTS price_snapshots (
    id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol    VARCHAR(20) NOT NULL,
    price     DECIMAL(14, 4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Szybkie pobranie najnowszej ceny dla symbolu (get_last_price).
CREATE INDEX IF NOT EXISTS idx_price_snapshots_symbol_timestamp
    ON price_snapshots (symbol, timestamp DESC);

-- Row Level Security WYŁĄCZONE — projekt backend-only, dostęp wyłącznie
-- z zaufanego środowiska. Patrz komentarz w 001_init.sql.
ALTER TABLE price_snapshots DISABLE ROW LEVEL SECURITY;
