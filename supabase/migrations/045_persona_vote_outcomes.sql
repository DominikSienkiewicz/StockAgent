-- =====================================================================
-- 045_persona_vote_outcomes.sql — surowe głosy person rady (WIA-51, WIA-46)
-- Uruchom PO 044_calibration_score_version.sql (ręczny workflow „DB Migrate”).
--
-- PO CO: RPC 015 (`persona_accuracy_weights`) i 018 (`persona_accuracy_stats`)
-- oceniają trafność głosu w SQL i dla HOLD biorą zawsze pasmo ±0.5%. Domena
-- (`sideways_tolerance_for`) daje krypto ±2.5%, więc HOLD na BTC/ETH był
-- oceniany 5× ostrzej niż predykcja tego samego cyklu (G45). Naprawa samego
-- 018 utrwaliłaby rozjazd z wagami z 015.
--
-- Ta funkcja zwraca SUROWE rozliczone głosy, bez oceny. Ocenę liczy domena
-- (`persona_vote_correct`, src/domain/persona_track_record.py) — jedną regułą
-- dla wag konsensusu i leaderboardu. Aplikacja czyta ją za flagą
-- `persona_vote_outcomes_enabled`; bez flagi dalej woła 015 i 018, które
-- zostają bez zmian (ścieżka sprzed migracji i powrót po wyłączeniu flagi).
--
-- `predicted_at` i `settled_at` (029) niosą oś czasu: wagi prequential
-- (WIA-46, `persona_accuracy_weights_as_of`) biorą tylko głosy rozliczone
-- przed chwilą predykcji. Okno `window_days` liczy się jak w 015 i 018 — od
-- NOW() po chwili predykcji; narzędzie ablacji podaje okno szersze o własne.
--
-- Wiersze posortowane deterministycznie, bo aplikacja czyta je stronami
-- (PostgREST obcina wynik RPC do `max_rows`).
--
-- Uprawnienia jak po 038: REVOKE EXECUTE od PUBLIC, GRANT dla service_role
-- w bloku DO z guardem pg_roles (kontener testów nie ma ról Supabase).
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE / GRANT / COMMENT można
-- powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION persona_vote_outcomes(window_days INT DEFAULT 90)
RETURNS TABLE (
    prediction_id UUID,
    investor_name TEXT,
    recommendation TEXT,
    confidence DOUBLE PRECISION,
    symbol TEXT,
    price_at_prediction NUMERIC,
    actual_price_after_12h NUMERIC,
    predicted_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        cv.prediction_id,
        cv.investor_name::TEXT,
        cv.recommendation::TEXT,
        cv.confidence::DOUBLE PRECISION,
        pl.symbol::TEXT,
        pl.price_at_prediction::NUMERIC,
        pl.actual_price_after_12h::NUMERIC,
        pl.timestamp AS predicted_at,
        pl.settled_at
    FROM council_votes cv
    JOIN prediction_logs pl ON pl.id = cv.prediction_id
    WHERE pl.actual_price_after_12h IS NOT NULL
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    ORDER BY pl.timestamp, cv.prediction_id, cv.investor_name;
$$;

REVOKE EXECUTE ON FUNCTION persona_vote_outcomes(INT) FROM PUBLIC;

-- Role API Supabase: publiczne bez EXECUTE (BEZ-05, jak 038), service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'REVOKE EXECUTE ON FUNCTION persona_vote_outcomes(INT) FROM %I', api_role
            );
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION persona_vote_outcomes(INT) TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION persona_vote_outcomes(INT) IS
    'WIA-51: rozliczone głosy person bez oceny — trafność liczy domena '
    '(pasmo HOLD wg klasy aktywa). Oś czasu predicted_at / settled_at dla wag '
    'prequential (WIA-46).';

COMMENT ON FUNCTION persona_accuracy_weights(INT) IS
    'Ścieżka sprzed WIA-51: HOLD zawsze ±0.5%, także na krypto. Za flagą '
    'persona_vote_outcomes_enabled aplikacja liczy wagi z persona_vote_outcomes.';

COMMENT ON FUNCTION persona_accuracy_stats(INT) IS
    'Ścieżka sprzed WIA-51: HOLD zawsze ±0.5%, także na krypto. Za flagą '
    'persona_vote_outcomes_enabled aplikacja liczy ranking z persona_vote_outcomes.';
