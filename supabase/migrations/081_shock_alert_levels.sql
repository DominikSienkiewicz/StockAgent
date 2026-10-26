-- =====================================================================
-- 081_shock_alert_levels.sql — drabina eskalacji alertów szoku (ALR-16).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: `UNIQUE(symbol, alert_date)` z 021 przepuszcza jeden alert symbolu dziennie.
-- Przy krachu od -4% do -12% użytkownik dostawał tylko pierwszy, łagodny sygnał.
-- Za flagą `shock_escalation_enabled` watch wysyła drugi alert dnia (szczebel 2),
-- gdy ruch sięgnie 2× progu szoku w tym samym kierunku albo odwróci się ponad próg
-- (`should_escalate` w src/domain/shock.py).
--
--   * `level` — szczebel drabiny; 1 = pierwszy alert dnia (także każdy wiersz sprzed
--     081 i każdy zapis bez klucza `level`), 2 = eskalacja. CHECK 1–2 to twardy sufit
--     „najwyżej dwa alerty symbolu dziennie”, także przy dwóch równoległych watchach;
--   * `UNIQUE(symbol, alert_date, level)` zastępuje `uq_shock_alerts_symbol_date` z 021:
--     jeden wiersz na szczebel. Flaga off zapisuje wyłącznie szczebel 1, więc dla niej
--     nic się nie zmienia.
--
-- Kontrakt dostarczenia zostaje at-least-once (ALR-14, ALR-23): watch wysyła push
-- PRZED zapisem wiersza, więc UNIQUE odrzuca drugi WIERSZ szczebla, nie drugi push.
-- Komentarz 021 opisuje odwrotną kolejność; 021 jest zaaplikowana i się jej nie edytuje.
--
-- Kolumna w istniejącej tabeli — RLS z 038 ją obejmuje, uprawnienia się nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS (CHECK w ADD COLUMN nie dubluje się przy
-- powtórce), nowe ograniczenie tylko gdy go brak, DROP CONSTRAINT IF EXISTS, COMMENT
-- można powtarzać. Nowe ograniczenie powstaje przed usunięciem starego, więc tabela
-- nie zostaje bez UNIQUE.
-- =====================================================================

ALTER TABLE shock_alerts
    ADD COLUMN IF NOT EXISTS level SMALLINT NOT NULL DEFAULT 1
        CONSTRAINT ck_shock_alerts_level CHECK (level BETWEEN 1 AND 2);

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_shock_alerts_symbol_date_level'
          AND conrelid = 'public.shock_alerts'::regclass
    ) THEN
        ALTER TABLE shock_alerts
            ADD CONSTRAINT uq_shock_alerts_symbol_date_level UNIQUE (symbol, alert_date, level);
    END IF;
END
$migration$;

ALTER TABLE shock_alerts DROP CONSTRAINT IF EXISTS uq_shock_alerts_symbol_date;

COMMENT ON TABLE shock_alerts IS
    'Rejestr alertów szoku poza cyklem: najwyżej dwa szczeble dziennie na symbol, '
    'UNIQUE(symbol, alert_date, level) (081, wcześniej UNIQUE(symbol, alert_date) z 021). '
    'Zapis po wysyłce pusha (at-least-once): UNIQUE odrzuca drugi wiersz, nie drugi push.';
COMMENT ON COLUMN shock_alerts.alert_date IS
    'Dzień alertu — razem ze szczeblem (`level`, 081) klucz debounce: najwyżej jeden '
    'wiersz na szczebel symbolu dziennie.';
COMMENT ON COLUMN shock_alerts.level IS
    'ALR-16: szczebel drabiny eskalacji — 1 = pierwszy alert dnia, 2 = eskalacja '
    '(ruch ≥ 2× progu szoku w tym samym kierunku albo odwrócenie ponad próg).';
