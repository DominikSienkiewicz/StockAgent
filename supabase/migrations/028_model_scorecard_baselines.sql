-- =====================================================================
-- 028_model_scorecard_baselines.sql — mocniejsze baseline'y karty modelu (ML-10)
-- Uruchom PO 027_feature_set_version.sql (ręczny workflow „DB Migrate").
--
-- PO CO: bramka walk-forward porównywała kandydata wyłącznie z persystencją
-- „zwrot = 0". XGBoost dostaje `llm_trend_signal` jako cechę, więc kandydat,
-- który bije zero, ale nie bije przeskalowanego SAMEGO sygnału LLM, tylko
-- kosztownie przepisuje LLM. Bramka (domena: `summarize_folds`) wymaga teraz
-- pobicia najsilniejszego z trzech baseline'ów z tym samym marginesem
-- względnym 2%:
--   * `baseline_rmse`       — „zwrot = 0" (migracja 019, bez zmian);
--   * `drift_baseline_rmse` — stały średni target części treningowej foldu;
--   * `llm_baseline_rmse`   — `llm_trend_signal` × średni |target| części
--                             treningowej foldu.
-- Oba nowe baseline'y biorą skalę WYŁĄCZNIE z części treningowej foldu — bez
-- podglądu walidacji. Kolumny trzymają średnią po foldach, jak `baseline_rmse`.
--
-- NULLABLE: wiersze sprzed tej migracji i przebiegi bez metryk (skipped) nie
-- mają tych wartości; Slow Loop zapisuje je tylko za flagą
-- `model_scorecard_baselines_enabled` (przed migracją nowy klucz → PGRST204,
-- a zapis karty jest best-effort i po cichu by przepadł).
-- Idempotentne: ADD COLUMN IF NOT EXISTS.
-- =====================================================================

ALTER TABLE model_scorecards
    -- Średni RMSE baseline'u „dryf średniej" na foldach walk-forward.
    ADD COLUMN IF NOT EXISTS drift_baseline_rmse DOUBLE PRECISION,
    -- Średni RMSE baseline'u „sam LLM" na foldach walk-forward.
    ADD COLUMN IF NOT EXISTS llm_baseline_rmse DOUBLE PRECISION;

COMMENT ON COLUMN model_scorecards.drift_baseline_rmse IS
    'ML-10: średni RMSE baseline''u dryfu (średni target części treningowej foldu); NULL = brak metryki.';
COMMENT ON COLUMN model_scorecards.llm_baseline_rmse IS
    'ML-10: średni RMSE baseline''u „sam LLM" (llm_trend_signal × średni |target| treningu); NULL = brak metryki.';
