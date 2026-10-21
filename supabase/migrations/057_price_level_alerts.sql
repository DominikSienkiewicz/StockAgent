-- =====================================================================
-- 057_price_level_alerts.sql — alerty poziomów cenowych ustawiane przez
-- użytkownika (ALR-11).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: „daj znać, gdy NVDA przebije 150” — watch (main_watch.py) ma darmowe
-- ceny i kanały push, ale nie miał gdzie trzymać poziomów. Jeden wiersz = jeden
-- poziom: symbol, cena, kierunek (ABOVE — przebicie w górę, BELOW — spadek do
-- poziomu) i opcjonalna notatka do treści pusha.
--
-- `last_seen_price` to poprzedni odczyt watcha: domena (`price_levels.crossed`)
-- odpala alert przy PRZEKROCZENIU poziomu między dwoma odczytami, a watch nie
-- zapisuje `price_snapshots` (kontrakt Fast Loopa). Po trafieniu watch ustawia
-- `active = false` i `triggered_at` — alert odpala raz.
--
-- Źródło startowe: lista `price_alerts` w config.toml; watch dopisuje brakujące
-- poziomy upsertem z ON CONFLICT DO NOTHING po `UNIQUE(symbol, level, direction)`,
-- więc poziom wyłączony po trafieniu nie wraca. Odczyt i zapis za flagą
-- `price_level_alerts_enabled` (przed migracją tabeli nie ma).
--
-- Uprawnienia jak po 038: ENABLE ROW LEVEL SECURITY bez polityk — `anon`
-- i `authenticated` nie widzą żadnego wiersza, watch pisze kluczem service_role.
--
-- Idempotentne: CREATE TABLE/INDEX IF NOT EXISTS, ENABLE RLS i COMMENT można
-- powtarzać.
-- =====================================================================

CREATE TABLE IF NOT EXISTS price_level_alerts (
    id               BIGSERIAL PRIMARY KEY,
    symbol           TEXT NOT NULL,
    level            NUMERIC NOT NULL CONSTRAINT price_level_alerts_level_positive
                         CHECK (level > 0),
    direction        TEXT NOT NULL CONSTRAINT price_level_alerts_direction_known
                         CHECK (direction IN ('ABOVE', 'BELOW')),
    note             TEXT,
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_price  NUMERIC,
    triggered_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT price_level_alerts_unique_level UNIQUE (symbol, level, direction)
);

-- Watch czyta wyłącznie aktywne alerty.
CREATE INDEX IF NOT EXISTS price_level_alerts_active_idx
    ON price_level_alerts (symbol) WHERE active;

ALTER TABLE price_level_alerts ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE price_level_alerts IS
    'Alerty poziomów cenowych użytkownika (ALR-11): watch odpala push przy przekroczeniu poziomu od last_seen_price i wyłącza alert po trafieniu.';

COMMENT ON COLUMN price_level_alerts.last_seen_price IS
    'Poprzedni odczyt ceny watcha — punkt odniesienia przekroczenia (watch nie zapisuje price_snapshots).';
