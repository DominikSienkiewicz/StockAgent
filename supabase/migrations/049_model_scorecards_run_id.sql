-- =====================================================================
-- 049_model_scorecards_run_id.sql — identyfikator przebiegu Slow Loopa
-- w karcie modelu (ML-24).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: karta w mailu pokazuje NAJNOWSZY przebieg treningu, a przebieg
-- rozpoznawała dotąd po przerwie czasowej między wierszami (> 2 h, ML-03).
-- Ręczny re-run kilka minut po cronie sklejał się z nim w jeden „przebieg”
-- (dwa wiersze ALL, werdykt z obu). `main_trainer` losuje teraz jeden uuid4
-- na przebieg i zapisuje go w każdym wierszu karty (per symbol i model
-- wspólny ALL); raport grupuje po nim, a dla historii bez `run_id` zostaje
-- heurystyka przerwy.
--
-- Zapis kolumny za flagą `model_scorecard_run_id_enabled` (domyślnie off):
-- przed tą migracją nowy klucz to PGRST204 i przepadły wiersz karty.
-- Stare wiersze zostają z NULL — przebiegów sprzed migracji nie da się
-- odtworzyć z bazy.
--
-- Uprawnienia: tabela ma RLS od 038; nowa kolumna i indeks tego nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS, COMMENT.
-- =====================================================================

ALTER TABLE model_scorecards
    ADD COLUMN IF NOT EXISTS run_id UUID;

-- Raport czyta wiersze najnowszego `run_id`; deadman i karta sortują po czasie.
CREATE INDEX IF NOT EXISTS idx_model_scorecards_run_id
    ON model_scorecards (run_id);

COMMENT ON COLUMN model_scorecards.run_id IS
    'Przebieg Slow Loopa (uuid4 losowany raz na przebieg w main_trainer), wspólny dla wierszy per symbol i ALL (ML-24). NULL = wiersz sprzed migracji 049 albo flaga off.';
