-- =====================================================================
-- 032_move_interval.sql — przedział ruchu z conformal prediction (ML-34)
-- Uruchom PO 031_feature_snapshots.sql (ręczny workflow „DB Migrate").
--
-- PO CO: produkt podawał gołą cenę docelową (przy cold-starcie równą bieżącej)
-- bez niepewności. Fast Loop liczy w każdym cyklu za bramką przedział 80%
-- zwrotu do rozliczenia (split-conformal + ACI, `src/domain/conformal.py`)
-- i pokazuje go w mailu bez tej migracji. Ta migracja pozwala go ZAPISAĆ, żeby
-- Track Record porównał pokrycie zrealizowane z nominalnym 80%.
--
-- Kolumny trzymają CENY (granice przedziału przeliczone z ceny predykcji), w tej
-- samej skali co `predicted_target_price` i `actual_price_after_12h`: pokrycie
-- to `interval_low <= actual_price_after_12h <= interval_high`.
--
-- UCZCIWOŚĆ: gwarancja pokrycia zakłada wymienność, której rynek nie spełnia;
-- ACI łagodzi zmianę rozkładu, ale nie gwarantuje pokrycia. Stąd pomiar.
--
-- BEZ BACKFILLU: przedziałów starych predykcji nikt nie policzył w chwili
-- predykcji. Stare wiersze i zapis bez flagi zostają NULL.
--
-- Zapisuje Fast Loop za flagą `interval_coverage_enabled` (przed migracją nowy
-- klucz → PGRST204 i śmierć zapisu całej predykcji).
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- Dolna granica przedziału 80% (cena); NULL = brak przedziału.
    ADD COLUMN IF NOT EXISTS interval_low  DECIMAL(14, 4),
    -- Górna granica przedziału 80% (cena); NULL = brak przedziału.
    ADD COLUMN IF NOT EXISTS interval_high DECIMAL(14, 4);

COMMENT ON COLUMN prediction_logs.interval_low IS
    'Dolna granica przedziału 80% ruchu do rozliczenia (ML-34, conformal + ACI); NULL = brak.';
COMMENT ON COLUMN prediction_logs.interval_high IS
    'Górna granica przedziału 80% ruchu do rozliczenia (ML-34, conformal + ACI); NULL = brak.';
