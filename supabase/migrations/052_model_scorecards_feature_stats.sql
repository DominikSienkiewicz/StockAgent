-- =====================================================================
-- 052_model_scorecards_feature_stats.sql — rozkład cech treningu w karcie
-- modelu (ML-21).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: `XGBoostAdapter.train()` liczył `feature_stats` (mean/std), ale
-- `save_model_scorecard` ich nie zapisywał, więc „drift-detector” istniał
-- tylko w docstringu. Teraz trening liczy per cecha:
--   mean, std               — jak dawniej,
--   edges, fractions        — biny kwantylowe (decyle) CAŁEJ próbki treningu:
--                             odniesienie do porównań z późniejszym rozkładem,
--   psi_recent_30d          — Population Stability Index ostatnich 30 dni próbki
--                             względem jej starszej części (`src/domain/drift.py`);
--                             NULL, gdy którakolwiek część ma < 20 wartości.
-- Karta w mailu ostrzega, gdy PSI > 0.25.
--
-- Zapis kolumny za flagą `model_scorecard_feature_stats_enabled` (domyślnie off):
-- przed tą migracją nowy klucz to PGRST204 i przepadły wiersz karty. Stare
-- wiersze zostają z NULL — rozkładu z dawnych treningów nie da się odtworzyć.
--
-- Uprawnienia: tabela ma RLS od 038; nowa kolumna tego nie zmienia.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, COMMENT.
-- =====================================================================

ALTER TABLE model_scorecards
    ADD COLUMN IF NOT EXISTS feature_stats JSONB;

COMMENT ON COLUMN model_scorecards.feature_stats IS
    'Rozkład cech treningu per cecha: mean, std, edges/fractions (biny kwantylowe całej próbki), psi_recent_30d (PSI ostatnich 30 dni vs starsza część próbki) — ML-21. NULL = wiersz sprzed migracji 052 albo flaga off.';
