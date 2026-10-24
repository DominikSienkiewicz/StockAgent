-- =====================================================================
-- 064_model_scorecards_provenance.sql — proweniencja treningu w karcie
-- modelu (ML-22).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: nie dało się odpowiedzieć, na jakich danych powstał model, który
-- wydał daną prognozę — pierwsze pytanie recenzenta ML i warunek uczciwego
-- championa/challengera. Każdy trening (model wspólny ALL, ewaluacje per
-- symbol, karty ALL_FREE/ALL_LLM) zapisuje teraz:
--   dataset_sha256      — sha256 wierszy treningu (chwila, target, cechy),
--                         niezależny od ich kolejności (`dataset_fingerprint`);
--   data_start/data_end — zakres znaczników czasu próbki (NULL bez znaczników);
--   n_per_symbol        — liczba wierszy treningu per symbol (JSONB);
--   training_params     — parametry XGBoost z jawnym `random_state` (JSONB);
--   library_version     — np. „xgboost 3.2.0” (inna wersja = inny booster);
--   feature_set_version — wersja zestawu cech próbki (NULL = cała historia);
--   git_sha             — commit kodu przebiegu (`GITHUB_SHA`, lokalnie „local”).
-- Ten sam odcisk i parametry odtwarzają booster: `main_tools.py reproduce-training`.
-- Plik wag dostaje te pola także w sidecarze `.json` (bez migracji i bez flagi).
--
-- Zapis kolumn za flagą `model_scorecard_provenance_enabled` (domyślnie off):
-- przed tą migracją nowe klucze to PGRST204 i przepadły wiersz karty. Stare
-- wiersze i pominięte treningi zostają z NULL.
--
-- Uprawnienia: tabela ma RLS od 038; nowe kolumny tego nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS (CHECK w ADD COLUMN nie dubluje się
-- przy powtórce), COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE model_scorecards
    ADD COLUMN IF NOT EXISTS dataset_sha256 TEXT
        CHECK (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS data_start TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS data_end TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS n_per_symbol JSONB,
    ADD COLUMN IF NOT EXISTS training_params JSONB,
    ADD COLUMN IF NOT EXISTS library_version TEXT,
    ADD COLUMN IF NOT EXISTS feature_set_version INTEGER,
    ADD COLUMN IF NOT EXISTS git_sha TEXT;

COMMENT ON COLUMN model_scorecards.dataset_sha256 IS
    'ML-22: sha256 (hex) wierszy treningu (chwila UTC, target, cechy), niezależny od kolejności wierszy. NULL = trening się nie odbył, wiersz sprzed 064 albo flaga off.';

COMMENT ON COLUMN model_scorecards.data_start IS
    'ML-22: najstarszy znacznik czasu próbki treningu. NULL = próbka bez znaczników.';

COMMENT ON COLUMN model_scorecards.data_end IS
    'ML-22: najnowszy znacznik czasu próbki treningu. NULL = próbka bez znaczników.';

COMMENT ON COLUMN model_scorecards.n_per_symbol IS
    'ML-22: liczba wierszy treningu per symbol, np. {"AAPL": 40, "BTC": 20}.';

COMMENT ON COLUMN model_scorecards.training_params IS
    'ML-22: parametry XGBoost treningu z jawnym random_state — z nimi refit odtwarza booster.';

COMMENT ON COLUMN model_scorecards.library_version IS
    'ML-22: wersja biblioteki modelu, np. "xgboost 3.2.0".';

COMMENT ON COLUMN model_scorecards.feature_set_version IS
    'ML-22: wersja zestawu cech próbki treningu (DAT-09, ML-31). NULL = cała historia.';

COMMENT ON COLUMN model_scorecards.git_sha IS
    'ML-22: commit kodu przebiegu Slow Loopa (GITHUB_SHA; "local" poza GitHub Actions).';
