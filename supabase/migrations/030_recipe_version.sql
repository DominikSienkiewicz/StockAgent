-- =====================================================================
-- 030_recipe_version.sql — wersja przepisu predykcji (WIA-20)
-- Uruchom PO 029_prediction_settled_at.sql (ręczny workflow „DB Migrate").
--
-- PO CO: track record mieszał wyniki różnych przepisów — zmiana promptu, nowy
-- model, nowe persony rady, inne progi albo nowe wagi XGBoost. Bez wersji nie
-- da się odróżnić umiejętności od zmiany przepisu, porównać „przed i po” ani
-- policzyć liczby prób do Deflated Sharpe. Ta migracja dodaje
-- `prediction_logs.recipe_version`: sha256 składników przepisu (provider
-- i model głównego LLM i rady, hash szablonu promptu, hash zestawu person,
-- hash niesekretnego configu, wersja wag ML). Commit i przebieg (`git_sha`,
-- `run_id`) są w kwicie v2 (`decision_receipts`), ale nie w tej wersji —
-- ten sam przepis w kolejnych przebiegach to jeden segment.
--
-- BEZ BACKFILLU: składników przepisu starych wierszy nie da się odtworzyć
-- z bazy (prompt i persony żyją w gicie, config bywał nadpisywany z env).
-- Stare wiersze i zapis bez flagi zostają NULL („przepis nieznany”).
--
-- Stempluje Fast Loop przy zapisie predykcji za flagą `recipe_version_enabled`
-- (przed migracją nowy klucz → PGRST204 i śmierć zapisu całej predykcji).
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- sha256 przepisu predykcji (WIA-20); NULL = przepis nieznany.
    ADD COLUMN IF NOT EXISTS recipe_version TEXT;

-- Segmentacja track recordu po przepisie (`src.tools.evaluate --by-recipe`).
CREATE INDEX IF NOT EXISTS idx_prediction_logs_recipe_version
    ON prediction_logs (recipe_version)
    WHERE recipe_version IS NOT NULL;

COMMENT ON COLUMN prediction_logs.recipe_version IS
    'Wersja przepisu predykcji (WIA-20): sha256 modeli LLM, szablonu promptu, person, configu i wag ML; NULL = nieznana.';
