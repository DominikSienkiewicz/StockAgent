-- =====================================================================
-- 041_cycle_runs.sql — historia przebiegów pętli: jeden wiersz na przebieg
-- (NIEZ-29). Uruchom PO 040_llm_calls.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: nie było śladu, czy cykl w ogóle ruszył, ile symboli zapisał i czy
-- raport wyszedł — mail mógł nie wyjść, a log runu GitHub Actions znika po 90
-- dniach. Ta tabela trzyma przebieg jako liczby, do deadmana, raportów i
-- rekoncyliacji kosztów z `llm_calls`.
--
-- Kolumny:
--   loop        — pętla: 'fast' (dziś pisze tylko Fast Loop), 'slow', 'watch';
--   run_id      — przebieg (`RunContext.run_id`: GITHUB_RUN_ID albo 'local') —
--                 ten sam identyfikator co `llm_calls.run_id`, więc koszt
--                 przebiegu łączy się z nim po `run_id`;
--   run_attempt — GITHUB_RUN_ATTEMPT: ponowienie runu ma ten sam `run_id`
--                 i kolejny numer próby; NULL poza GitHub Actions;
--   started_at, duration_s — start cyklu i czas całego przebiegu;
--   eligible, saved, failures — symbole cyklu, zapisane predykcje, porażki;
--   paid_calls  — płatne wywołania per źródło cennika (`{"llm": 7, ...}`);
--   report_delivered — czy mail wyszedł; NULL = przebieg bez kanału e-mail.
-- Świadomie BEZ symboli, treści raportu i błędów: same liczby przebiegu.
--
-- Zapis jednym insertem na końcu Fast Loopa za flagą `cycle_runs_enabled`
-- (przed migracją tabeli nie ma — każdy zapis kończyłby się błędem w logu).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta. Granty dla anon / authenticated
-- odbierają domyślne uprawnienia ustawione w 038.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS cycle_runs (
    id               BIGSERIAL PRIMARY KEY,
    loop             TEXT NOT NULL CONSTRAINT cycle_runs_loop_known
                         CHECK (loop IN ('fast', 'slow', 'watch')),
    run_id           TEXT NOT NULL,
    run_attempt      INTEGER CONSTRAINT cycle_runs_run_attempt_positive
                         CHECK (run_attempt >= 1),
    started_at       TIMESTAMPTZ NOT NULL,
    duration_s       DOUBLE PRECISION NOT NULL CONSTRAINT cycle_runs_duration_non_negative
                         CHECK (duration_s >= 0),
    eligible         INTEGER NOT NULL CONSTRAINT cycle_runs_eligible_non_negative
                         CHECK (eligible >= 0),
    saved            INTEGER NOT NULL CONSTRAINT cycle_runs_saved_non_negative
                         CHECK (saved >= 0),
    failures         INTEGER NOT NULL CONSTRAINT cycle_runs_failures_non_negative
                         CHECK (failures >= 0),
    paid_calls       JSONB NOT NULL DEFAULT '{}'::jsonb,
    report_delivered BOOLEAN,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Deadman i raporty czytają najnowsze przebiegi („kiedy ostatnio ruszył”).
CREATE INDEX IF NOT EXISTS idx_cycle_runs_started_at
    ON cycle_runs (started_at DESC);

ALTER TABLE cycle_runs ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE cycle_runs IS
    'Historia przebiegów: jeden wiersz na przebieg pętli (liczby, płatne wywołania, stan maila), bez symboli (NIEZ-29).';
COMMENT ON COLUMN cycle_runs.run_id IS
    'RunContext.run_id (GITHUB_RUN_ID albo local) — łączy się z llm_calls.run_id.';
COMMENT ON COLUMN cycle_runs.run_attempt IS
    'GITHUB_RUN_ATTEMPT — rozróżnia ponowienia tego samego runu; NULL poza GitHub Actions.';
COMMENT ON COLUMN cycle_runs.report_delivered IS
    'Czy mail cyklu wyszedł; NULL = przebieg bez kanału e-mail.';
