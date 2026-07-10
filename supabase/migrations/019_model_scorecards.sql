-- =====================================================================
-- 019_model_scorecards.sql — jawny scorecard walk-forward modelu (#12)
-- Uruchom w Supabase SQL Editor PO 018_persona_track_record.sql.
--
-- Karta kondycji modelu: raport pokazuje NIEWIDZIALNY DZIŚ dowód rzetelności —
-- bramkę "nie shipuj modelu gorszego od baseline'u" z `xgboost_local.py`. Każdy
-- przebieg treningu (Slow Loop) zapisuje tu jeden wiersz: jak kandydat wypadł na
-- foldach walk-forward względem naiwnego baseline'u persystencji ("zwrot = 0")
-- i czy został wdrożony, czy odrzucony.
--
-- KOREKTA KRYTYCZNA WERYFIKATORA: trening jest PER-SYMBOL — ~43 przebiegi
-- nadpisują jeden plik .ubj, więc bez kolumny `symbol` scorecardy są
-- nierozróżnialne (nie wiadomo, który symbol zaakceptowano, a który odrzucono).
-- Kolumna `symbol` jest wymaganiem, nie opcją.
--
-- Nazwy kolumn metryk 1:1 z kluczami wyniku `XGBoostAdapter.train()`:
--   candidate_holdout_rmse, baseline_rmse,
--   candidate_holdout_directional_hit_rate, n_folds, status.
-- status ∈ {'trained_successfully', 'skipped_validation_failed'} — z train().
-- UWAGA copy raportu (METRIC_NOTE): raportowane metryki dotyczą KANDYDATA na
-- foldach; shipowany model jest refitowany na całości i NIE jest re-walidowany.
--
-- Idempotentne (IF NOT EXISTS).
-- =====================================================================

CREATE TABLE IF NOT EXISTS model_scorecards (
    id                                     BIGSERIAL PRIMARY KEY,
    -- Symbol, którego dotyczy ten przebieg treningu (per-symbol training).
    symbol                                 TEXT NOT NULL,
    -- Werdykt bramki z train(): wdrożony vs odrzucony (gorszy od baseline'u).
    status                                 TEXT NOT NULL,
    -- Średni RMSE kandydata na holdoutach foldów walk-forward.
    candidate_holdout_rmse                 DOUBLE PRECISION,
    -- Średni RMSE naiwnego baseline'u persystencji ("zwrot = 0").
    baseline_rmse                          DOUBLE PRECISION,
    -- Średnia trafność KIERUNKU zwrotu kandydata na foldach ∈ [0, 1].
    candidate_holdout_directional_hit_rate DOUBLE PRECISION,
    -- Liczba foldów walk-forward, po których uśredniono metryki.
    n_folds                                INTEGER,
    -- Liczba próbek treningowych symbolu w tym przebiegu (kontekst metryk).
    n_samples                              INTEGER,
    -- Czas przebiegu treningu (best-effort zapis w main_trainer).
    timestamp                              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Sekcja raportu pobiera NAJNOWSZY scorecard per symbol → indeks po
-- (symbol, timestamp DESC) obsługuje ten wzorzec dostępu wprost.
CREATE INDEX IF NOT EXISTS idx_model_scorecards_symbol_timestamp
    ON model_scorecards (symbol, timestamp DESC);

COMMENT ON TABLE model_scorecards IS
    'Scorecardy walk-forward per symbol: metryki kandydata vs baseline + werdykt bramki.';
COMMENT ON COLUMN model_scorecards.symbol IS
    'Symbol trenowany w tym przebiegu — trening jest per-symbol (~43 przebiegi).';
COMMENT ON COLUMN model_scorecards.status IS
    'Werdykt train(): trained_successfully (wdrożony) / skipped_validation_failed (odrzucony).';
