-- =====================================================================
-- 051_trend_probabilities.sql — rozkład P(BULLISH/BEARISH/SIDEWAYS)
-- predykcji (ML-29).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: jedna pewność dla wybranej klasy nie mówi, czy model „prawie” wybrał
-- inną, a właściwe reguły punktacji (RPS, 3-klasowy Brier) wymagają pełnego
-- rozkładu. Za flagą `trend_probabilities_enabled` prompt prosi o rozkład,
-- domena (`ProbabilityForecast`) normalizuje go do sumy 1, trend to argmax,
-- a `confidence_score` to max p. Ta kolumna przechowuje znormalizowany rozkład:
-- `{"BULLISH": 0.2, "BEARISH": 0.1, "SIDEWAYS": 0.7}`.
--
-- NULL = predykcja sprzed migracji, flaga off albo LLM nie podał poprawnego
-- rozkładu (trend i pewność wtedy ze starych pól odpowiedzi). Zapis za flagą:
-- przed migracją nowy klucz to PGRST204 i śmierć zapisu całej predykcji.
-- Flaga zmienia prompt, więc należy do przepisu (`config_hash`) — track record
-- dzieli się po `recipe_version`, a rozkład pewności przed i po zmianie formatu
-- się nie miesza.
--
-- Uprawnienia: tabela ma RLS od 038; nowa kolumna tego nie zmienia.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, COMMENT.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS trend_probabilities JSONB;

COMMENT ON COLUMN prediction_logs.trend_probabilities IS
    'Znormalizowany rozkład P(BULLISH/BEARISH/SIDEWAYS) z odpowiedzi LLM (suma 1, ML-29); trend = argmax, confidence_score = max p. NULL = przed migracją 051, flaga off albo brak poprawnego rozkładu.';
