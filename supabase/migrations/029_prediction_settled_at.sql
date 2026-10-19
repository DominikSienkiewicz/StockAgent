-- =====================================================================
-- 029_prediction_settled_at.sql — chwila rozliczenia predykcji (WIA-19)
-- Uruchom PO 028_model_scorecard_baselines.sql (ręczny workflow „DB Migrate").
--
-- PO CO: okna „zamknięte w ostatnich N” (sekcja 24 h w mailu, recap
-- tygodniowy) filtrowały po `timestamp`, czyli po dacie UTWORZENIA predykcji.
-- Predykcja z piątku rozliczona w poniedziałek ma już ponad 24 h, więc nie
-- trafiała do sekcji maila, a w niedzielnym recapie była jeszcze otwarta —
-- tydzień później wypadała już z okna 7 dni. Ta migracja dodaje
-- `prediction_logs.settled_at`, stemplowane przez trigger w chwili, gdy
-- `actual_price_after_12h` przechodzi z NULL na wartość. Okna po `settled_at`
-- czyta aplikacja za flagą `settled_at_windows_enabled` (przed migracją
-- filtr po nieistniejącej kolumnie → 400).
--
-- BACKFILL (PRZYBLIŻENIE — prawdziwa chwila rozliczenia sprzed triggera nie
-- jest nigdzie zapisana):
--   * Cykl, który rozlicza predykcję (`reflect_node`), zapisuje w węźle
--     terminalnym snapshot ceny PO TEJ SAMEJ cenie (`state["current_price"]`),
--     kilka sekund po rozliczeniu. Obie kolumny to DECIMAL(14,4), więc
--     zaokrąglenie jest identyczne. Dlatego `settled_at` = `timestamp`
--     PIERWSZEGO snapshotu tego symbolu po cenie `actual_price_after_12h`,
--     zrobionego co najmniej godzinę po predykcji. Godzina odcina snapshot
--     cyklu, który predykcję ZAPISAŁ (gdy cena się nie ruszyła, ma tę samą
--     cenę), a jest dużo krótsza niż odstęp cykli (12 h) i próg
--     `reflection_min_age_hours` (6 h, od 48378c4 z 2026-06-05).
--   * Ograniczenie: gdy cena stała przez kilka cykli (weekend, święto) i cykl
--     rozliczał zaległość innej predykcji tego symbolu, dopasuje się wcześniejszy
--     cykl z tą samą ceną — błąd mieści się między utworzeniem a prawdziwym
--     rozliczeniem.
--   * Gdy takiego snapshotu nie ma (snapshot cyklu się nie zapisał, wiersz
--     sprzed 002, rozliczenie inną ceną niż snapshot, rozliczenie w ciągu
--     godziny przez nakładający się ręczny run sprzed progu wieku), zostaje
--     DOLNA GRANICA:
--     `settled_at = timestamp`. To nie jest chwila rozliczenia, tylko
--     najwcześniejsza możliwa — taki wiersz poznać po `settled_at = timestamp`.
--
-- TRIGGER: BEFORE UPDATE, WHEN (OLD … IS NULL AND NEW … IS NOT NULL) — stempel
-- powstaje raz. Ponowna ocena rozliczonego wiersza go nie przesuwa; ręczna
-- korekta `settled_at` osobnym UPDATE-em też przechodzi (warunek nie zachodzi).
-- Upsert (INSERT … ON CONFLICT DO UPDATE) przechodzi przez ten sam trigger.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, backfill tylko po NULL-ach,
-- CREATE OR REPLACE FUNCTION, DROP TRIGGER IF EXISTS + CREATE TRIGGER,
-- CREATE INDEX IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Chwila rozliczenia (WIA-19); NULL = predykcja jeszcze otwarta.
    ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ;

UPDATE prediction_logs AS p
SET settled_at = COALESCE(
    (
        SELECT s.timestamp
        FROM price_snapshots AS s
        WHERE s.symbol = p.symbol
          AND s.price = p.actual_price_after_12h
          AND s.timestamp >= p.timestamp + INTERVAL '1 hour'
        ORDER BY s.timestamp ASC
        LIMIT 1
    ),
    p.timestamp
)
WHERE p.actual_price_after_12h IS NOT NULL
  AND p.settled_at IS NULL;

CREATE OR REPLACE FUNCTION stamp_prediction_settled_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.settled_at := now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prediction_logs_settled_at ON prediction_logs;

CREATE TRIGGER trg_prediction_logs_settled_at
    BEFORE UPDATE ON prediction_logs
    FOR EACH ROW
    WHEN (OLD.actual_price_after_12h IS NULL AND NEW.actual_price_after_12h IS NOT NULL)
    EXECUTE FUNCTION stamp_prediction_settled_at();

-- Okna „rozliczone od X” (sekcja maila, recap) filtrują i sortują po settled_at.
CREATE INDEX IF NOT EXISTS idx_prediction_logs_settled_at
    ON prediction_logs (settled_at DESC)
    WHERE settled_at IS NOT NULL;

COMMENT ON COLUMN prediction_logs.settled_at IS
    'Chwila rozliczenia (trigger przy pierwszym actual_price_after_12h, 029). Backfill: snapshot cyklu rozliczenia; settled_at = timestamp oznacza dolną granicę (chwila nieznana).';
