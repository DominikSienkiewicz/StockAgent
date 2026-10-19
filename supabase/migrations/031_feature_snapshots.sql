-- =====================================================================
-- 031_feature_snapshots.sql — wektor cech point-in-time w każdym cyklu (ML-14)
-- Uruchom PO 030_recipe_version.sql (ręczny workflow „DB Migrate").
--
-- PO CO: `ml_feature_store` powstaje z `prediction_logs`, czyli wyłącznie
-- z cykli, które przeszły bramkę volatility. Model uczy się na wycinku rynku
-- (dni z dużym ruchem), a próbki na symbol zbierają się miesiącami. Ta tabela
-- dostaje wiersz w KAŻDYM ukończonym cyklu — także ignored — z węzłów
-- terminalnych grafu, obok snapshotu ceny:
--   * cechy darmowe (cena, delta vs snapshot, próg efektywny, reżim i mnożniki
--     progu, Alpha Fusion, typ aktywa, werdykt wyceny) zawsze i wyłącznie ze
--     stanu cyklu — zero nowych wywołań zewnętrznych;
--   * cechy LLM (wektor `ML_FEATURE_COLUMNS` dokładnie w postaci podanej
--     modelowi, pewność LLM, flagi jakości) tylko w cyklu za bramką
--     (`gated = true`).
-- Trening czytający ten wektor nie odtwarza cech widokiem SQL, więc nie ma
-- skew „SQL w treningu vs Python w servingu”.
--
-- BEZ ETYKIETY: w chwili zapisu przyszłości nie ma. Zwrot dołącza się później
-- z `price_snapshots` po (symbol, as_of). BEZ BACKFILLU: historia nie ma
-- wektora z chwili predykcji — tabela startuje pusta.
--
-- Idempotencja jak 013: generowana kolumna `as_of_hour` + UNIQUE(symbol,
-- as_of_hour); adapter robi upsert `on_conflict="symbol,as_of_hour"`, więc
-- re-run w tej samej godzinie aktualizuje wiersz zamiast go duplikować.
--
-- Zapisuje Fast Loop za flagą `feature_snapshots_enabled` (przed migracją
-- tabeli nie ma → błąd PostgREST przy każdym cyklu).
--
-- Idempotentne: CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS.
-- =====================================================================

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    symbol              TEXT NOT NULL,
    -- Chwila cyklu (point-in-time). Stawia ją baza, jak w price_snapshots.
    as_of               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Klucz idempotencji; 3-argumentowy date_trunc jest IMMUTABLE (patrz 013).
    as_of_hour          TIMESTAMPTZ
        GENERATED ALWAYS AS (date_trunc('hour', as_of, 'UTC')) STORED,
    -- `FEATURE_SET_VERSION` z src/application/ml_features.py (DAT-09).
    feature_set_version INTEGER NOT NULL,
    -- Wektor cech: klucze darmowe zawsze, klucze LLM tylko gdy gated.
    features            JSONB NOT NULL,
    -- Czy cykl przeszedł bramkę volatility (raport selekcji: gated vs non-gated).
    gated               BOOLEAN NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_hour_uniq
    ON feature_snapshots (symbol, as_of_hour);

-- Okno czasowe przez wszystkie symbole (dołączanie etykiet, raport selekcji).
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_as_of
    ON feature_snapshots (as_of DESC);

-- Trening na bieżącej wersji cech, osobno wiersze gated i non-gated.
CREATE INDEX IF NOT EXISTS idx_feature_snapshots_version_gated
    ON feature_snapshots (feature_set_version, gated);

COMMENT ON TABLE feature_snapshots IS
    'Wektor cech point-in-time z każdego cyklu Fast Loopa (ML-14), bez etykiety.';
COMMENT ON COLUMN feature_snapshots.features IS
    'Cechy darmowe zawsze; wektor ML_FEATURE_COLUMNS, pewność i flagi jakości tylko gdy gated.';
COMMENT ON COLUMN feature_snapshots.gated IS
    'true = cykl przeszedł bramkę volatility i zapłacił za cechy LLM.';
