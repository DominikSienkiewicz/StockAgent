-- =====================================================================
-- 044_calibration_score_version.sql — wersja reguły confidence_calibration (WIA-37)
-- Uruchom PO 042_reference_predictions.sql (ręczny workflow „DB Migrate").
--
-- PO CO: sędzia kalibracji (#4, migracja 016) zapisywał per predykcję
-- `1 − |c − o|` — regułę NIEWŁAŚCIWĄ: przy prawdziwej trafności 0.6 zgłoszenie
-- c = 1.0 daje oczekiwane 0.6, a uczciwe c = 0.6 tylko 0.52, więc nagradza
-- przepewność. Nowa reguła to Brier `1 − (c − o)²` (właściwa). Obie trafiają
-- do tej samej kolumny, więc wiersz musi mówić, którą policzono:
--   1 = reguła liniowa (wszystko zapisane przed włączeniem flagi),
--   2 = reguła Briera (zapis za flagą `calibration_score_version_enabled`).
--
-- BACKFILL: każdy wiersz z oceną i bez wersji dostaje 1 — przed flagą kod
-- zapisywał wyłącznie regułę liniową, a wynik Briera idzie zawsze RAZEM
-- z wersją 2. Ponowny przebieg nie rusza wierszy z wersją. Wiersz oceniony
-- po migracji, ale jeszcze bez flagi, ma NULL — czytaj go jako 1.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS calibration_score_version SMALLINT
        CHECK (calibration_score_version IN (1, 2));

UPDATE prediction_logs
SET calibration_score_version = 1
WHERE confidence_calibration IS NOT NULL
  AND calibration_score_version IS NULL;

COMMENT ON COLUMN prediction_logs.calibration_score_version IS
    'WIA-37: reguła confidence_calibration — 1 = 1 − |c − o| (niewłaściwa), '
    '2 = 1 − (c − o)² (Brier). NULL przy ocenie = 1.';
