-- =====================================================================
-- 083_shadow_prediction_trend.sql — trend i pewność wariantu promptu albo modelu
-- LLM w `shadow_predictions` (EXP-26).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: zmianę promptu albo modelu da się uczciwie ocenić tylko równolegle
-- z produkcją, na tych samych wejściach (retro-eval ma lookahead, „przed i po”
-- miesza zmianę z rynkiem). Fast Loop pyta dodatkowo wariant dla stałej podpróbki
-- symboli za bramką volatility i zapisuje tu jego trend obok predykcji produkcyjnej
-- (ten sam `prediction_id`). Wariant nie wpływa na mail, radę ani `prediction_logs`;
-- rozlicza się tą samą regułą domeny co produkcja (`Prediction.is_trend_correct`).
--
-- Wiersz wariantu niesie `predicted_trend` (BULLISH/BEARISH/SIDEWAYS) i
-- `confidence_score` zamiast prognozy zwrotu albo P(wzrost) nogi ML (075): kolumny
-- nogi ML trendu nie wyrażą (SIDEWAYS nie ma odpowiednika w P(wzrost)).
-- `model_name` to nazwa wariantu z config.toml, `model_version` — sha256 jego
-- tożsamości (model, hash szablonu promptu, nadpisane opcje), jak CHECK z 075.
-- Odczyt nogi ML (ML-33) pomija takie wiersze: nie mają ani zwrotu, ani P(wzrost).
--
-- Zapis za flagą `shadow_prompt_variant_enabled` (przed migracją kolumn nie ma).
-- Tabela ma RLS od 075; nowe kolumny go dziedziczą, nowych obiektów nie ma.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, DROP CONSTRAINT IF EXISTS przed ADD
-- i COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS predicted_trend TEXT;
ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION;

ALTER TABLE shadow_predictions
    DROP CONSTRAINT IF EXISTS shadow_predictions_predicted_trend_known;
ALTER TABLE shadow_predictions
    ADD CONSTRAINT shadow_predictions_predicted_trend_known
    CHECK (predicted_trend IN ('BULLISH', 'BEARISH', 'SIDEWAYS'));

ALTER TABLE shadow_predictions
    DROP CONSTRAINT IF EXISTS shadow_predictions_confidence_unit;
ALTER TABLE shadow_predictions
    ADD CONSTRAINT shadow_predictions_confidence_unit
    CHECK (confidence_score >= 0 AND confidence_score <= 1);

-- Prognoza wiersza: zwrot albo P(wzrost) nogi ML (075) albo trend wariantu LLM.
ALTER TABLE shadow_predictions
    DROP CONSTRAINT IF EXISTS shadow_predictions_has_forecast;
ALTER TABLE shadow_predictions
    ADD CONSTRAINT shadow_predictions_has_forecast
    CHECK (predicted_return IS NOT NULL OR proba IS NOT NULL OR predicted_trend IS NOT NULL);

COMMENT ON COLUMN shadow_predictions.predicted_trend IS
    'Trend wariantu promptu albo modelu LLM w cieniu (EXP-26); NULL w wierszach nogi ML.';

COMMENT ON COLUMN shadow_predictions.confidence_score IS
    'Pewność [0, 1] wariantu LLM w cieniu po tej samej normalizacji co produkcja (EXP-26).';
