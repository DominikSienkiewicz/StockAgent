-- =====================================================================
-- 037_experiment_arms.sql — ramiona eksperymentów w pętli (EXP-05)
-- Uruchom PO 036_fx_rates.sql (ręczny workflow „DB Migrate").
--
-- PO CO: eksperyment w pętli (np. ślepa rada EXP-02, A/B pamięci EXP-03)
-- przydziela każdemu symbolowi-dniu ramię deterministycznie: sha256(id|symbol|
-- dzień UTC) mod liczba ramion. Żeby porównać ramiona na tym samym okresie,
-- ramię musi być zapisane przy predykcji. Kwit decyzyjny v2 niesie je już jako
-- opcjonalny klucz `decision_receipts->'experiment_arms'` (bez migracji, za
-- `receipts_enabled`); ta kolumna wystawia je wprost do zapytań SQL:
-- `prediction_logs.experiment_arms` = `{"EXP-02": "blind"}`, NULL = w cyklu
-- nie działał żaden eksperyment.
--
-- BACKFILL: z kwitu v2, gdy niesie obiekt `experiment_arms` (wiersze zapisane
-- przed migracją z aktywnym eksperymentem i włączonymi kwitami). Bez kwitu
-- ramienia nie da się odtworzyć: przydział zależy od definicji eksperymentu
-- z dnia cyklu, a ta żyje w `config.toml`, nie w bazie.
--
-- Zapisuje Fast Loop za flagą `experiment_arms_enabled` (przed migracją nowy
-- klucz → PGRST204 i śmierć zapisu całej predykcji). `evaluate --experiment`
-- przed migracją bierze ramiona z kwitu i drukuje komunikat o migracji.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, backfill tylko po NULL-ach.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Ramiona aktywnych eksperymentów `{id: ramię}` (EXP-05); NULL = brak eksperymentu.
    ADD COLUMN IF NOT EXISTS experiment_arms JSONB;

UPDATE prediction_logs
SET experiment_arms = decision_receipts -> 'experiment_arms'
WHERE experiment_arms IS NULL
  AND jsonb_typeof(decision_receipts -> 'experiment_arms') = 'object';

COMMENT ON COLUMN prediction_logs.experiment_arms IS
    'Ramiona aktywnych eksperymentów {id: ramię} (EXP-05). Backfill 037 z kwitu v2 (decision_receipts->experiment_arms); NULL = brak eksperymentu w cyklu.';
