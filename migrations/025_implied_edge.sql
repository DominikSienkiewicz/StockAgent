-- =====================================================================
-- 025_implied_edge.sql — edge predykcji vs implied move rynku opcji (#18)
-- Uruchom w Supabase SQL Editor PO 024_attestation.sql.
--
-- PO CO: quant chce wiedzieć nie "co przewiduje model", tylko "czy model wie
-- coś, czego rynek jeszcze nie wycenił". Repo ma obie strony równania —
-- XGBoost `predicted_target_price` (ruch modelu) i `OptionsFlowSnapshot.
-- implied_vol` (ruch wyceniony przez opcje) — i dotąd ich nie stykało. Domena
-- (`implied_edge.py`) liczy `edge_sigma` = o ile sigm ruch modelu odchyla się od
-- implied move opcji (np. "model +3.1%, opcje ±1.2% → edge 2.6σ"). Ta kolumna
-- utrwala sygnał, żeby po N tygodniach dało się policzyć hit-rate podzbioru
-- edge>1.5σ vs reszta = udokumentowana alfa albo uczciwy wynik negatywny.
--
-- NULLABLE: sygnał powstaje tylko za flagą `options_flow_enabled` i za bramką
-- volatility (fetch w predict_node); bez danych opcyjnych (Finnhub 403 → None,
-- graceful) `edge_sigma` zostaje NULL i cykl jest identyczny jak dziś.
-- DOUBLE PRECISION — domena liczy na float + math.sqrt (mieszanie z Decimal
-- wywala mypy), więc typ kolumny zgodny z typem domenowym.
-- Idempotentne: ADD COLUMN IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Dywergencja ruchu modelu od implied move opcji, w sigmach; NULL gdy brak
    -- danych opcyjnych lub flaga options_flow_enabled = false.
    ADD COLUMN IF NOT EXISTS edge_sigma DOUBLE PRECISION;

COMMENT ON COLUMN prediction_logs.edge_sigma IS
    'Edge predykcji vs implied move opcji w sigmach (#18); NULL gdy brak danych/flaga OFF.';
