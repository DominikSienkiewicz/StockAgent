-- =====================================================================
-- 075_shadow_predictions.sql — prognozy cienia nogi ML: champion i challengerzy
-- (ML-33).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: shipowany model XGBoost jest refitowany na całości i nie jest
-- re-walidowany, więc jedyna uczciwa ocena poza próbą to predykcje live. Fast Loop
-- zapisuje tu prognozę championa (tę, która poszła do maila) i do dwóch
-- challengerów z tymi samymi cechami — challengerzy nie wpływają na wynik. Slow Loop
-- porównuje parami (ta sama predykcja, po rozliczeniu w `prediction_logs`) regułą
-- domenową `should_promote` i raportuje werdykt w summary; promocję robi właściciel.
--
-- Jeden wiersz = jeden model dla jednej predykcji: `model_name` ('champion' albo
-- nazwa challengera z config.toml), `model_version` (sha256 pliku wag, jak
-- `model_version` przepisu, CI-13), `predicted_return` (zwrot 12 h regresora,
-- w surowych jednostkach) albo `proba` (P(zwrot > martwa strefa) klasyfikatora ML-30).
-- UNIQUE(prediction_id, model_name): re-run w tej samej godzinie nadpisuje
-- predykcję upsertem, więc wiersz cienia też idzie upsertem.
--
-- Zapis za flagą `shadow_predictions_enabled` (przed migracją tabeli nie ma).
--
-- Uprawnienia jak po 038: ENABLE ROW LEVEL SECURITY bez polityk — `anon`
-- i `authenticated` nie widzą żadnego wiersza, pętle piszą kluczem service_role.
--
-- Idempotentne: CREATE TABLE/INDEX IF NOT EXISTS, ENABLE RLS i COMMENT można
-- powtarzać.
-- =====================================================================

CREATE TABLE IF NOT EXISTS shadow_predictions (
    id                BIGSERIAL PRIMARY KEY,
    prediction_id     UUID NOT NULL REFERENCES prediction_logs(id) ON DELETE CASCADE,
    symbol            TEXT NOT NULL,
    model_name        TEXT NOT NULL CONSTRAINT shadow_predictions_model_name_present
                          CHECK (model_name <> ''),
    model_version     TEXT NOT NULL CONSTRAINT shadow_predictions_model_version_sha256
                          CHECK (model_version ~ '^[0-9a-f]{64}$'),
    predicted_return  DOUBLE PRECISION,
    proba             DOUBLE PRECISION CONSTRAINT shadow_predictions_proba_unit
                          CHECK (proba >= 0 AND proba <= 1),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT shadow_predictions_has_forecast
        CHECK (predicted_return IS NOT NULL OR proba IS NOT NULL),
    CONSTRAINT shadow_predictions_one_per_model UNIQUE (prediction_id, model_name)
);

-- Slow Loop czyta okno ostatnich dni.
CREATE INDEX IF NOT EXISTS shadow_predictions_created_at_idx
    ON shadow_predictions (created_at);

ALTER TABLE shadow_predictions ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE shadow_predictions IS
    'Prognozy cienia nogi ML (ML-33): champion i challengerzy dla tej samej predykcji; challengerzy nie wpływają na wynik, Slow Loop porównuje je parami po rozliczeniu.';

COMMENT ON COLUMN shadow_predictions.model_version IS
    'sha256 pliku wag modelu (jak model_version przepisu predykcji, CI-13).';

COMMENT ON COLUMN shadow_predictions.predicted_return IS
    'Zwrot 12 h z regresora w surowych jednostkach (model w jednostkach sigma już przemnożony przez sigma).';

COMMENT ON COLUMN shadow_predictions.proba IS
    'P(zwrot 12 h > martwa strefa protokołu) z klasyfikatora kierunku (ML-30).';
