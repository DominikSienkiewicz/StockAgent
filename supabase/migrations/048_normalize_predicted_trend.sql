-- =====================================================================
-- 048_normalize_predicted_trend.sql — kanoniczny `predicted_trend` i CHECK,
-- który blokuje kolejne zatrucia (ML-19).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: widok `ml_feature_store` mapuje trend na `llm_trend_signal` porównaniem
-- `= 'BULLISH'` / `= 'BEARISH'`, a track record liczy trafność kierunku po tej
-- samej kolumnie. ' bullish ' albo 'Bearish' z czasów przed kanonizacją (ML-07)
-- wypadały w cichy SIDEWAYS (sygnał 0), a nic w schemacie nie broniło przed
-- kolejnym takim zapisem. Graf zapisuje dziś wyłącznie BULLISH / BEARISH /
-- SIDEWAYS (`normalize_trend` w domenie), więc CHECK pilnuje tylko, żeby tak
-- zostało — także przy ręcznych zapisach i przyszłych błędach.
--
-- KOLEJNOŚĆ (pułapka): najpierw UPDATE do UPPER(TRIM(...)), potem CHECK jako
-- NOT VALID. CHECK bez NOT VALID albo przed UPDATE-em sprawdziłby całą historię
-- i wywalił migrację na pierwszym wierszu spoza listy. NOT VALID nie skanuje
-- istniejących wierszy, ale sprawdza każdy INSERT i każdy UPDATE — także UPDATE
-- starego wiersza: wiersz z wartością spoza listy (np. 'UP') po migracji nie
-- przyjmie zapisu (rozliczenie, kalibracja), dopóki ktoś nie poprawi jego trendu.
-- Liczbę takich wierszy migracja podaje w NOTICE; gdy jest 0, właściciel może
-- dokończyć `ALTER TABLE prediction_logs VALIDATE CONSTRAINT
-- prediction_logs_predicted_trend_known`. NULL przechodzi (wiersz bez trendu).
--
-- Trigger `trg_prediction_logs_settled_at` (029) nie rusza: UPDATE nie zmienia
-- `actual_price_after_12h`.
--
-- Idempotentne: UPDATE dotyka tylko wierszy, które się zmieniają; CHECK powstaje
-- tylko, gdy go nie ma (powtórka nie zastępuje już zwalidowanego ograniczenia).
-- =====================================================================

UPDATE prediction_logs
SET predicted_trend = UPPER(TRIM(predicted_trend))
WHERE predicted_trend IS NOT NULL
  AND predicted_trend <> UPPER(TRIM(predicted_trend));

DO $migration$
DECLARE
    unknown_rows bigint;
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'prediction_logs'::regclass
          AND conname = 'prediction_logs_predicted_trend_known'
    ) THEN
        ALTER TABLE prediction_logs
            ADD CONSTRAINT prediction_logs_predicted_trend_known
            CHECK (predicted_trend IN ('BULLISH', 'BEARISH', 'SIDEWAYS')) NOT VALID;
    END IF;

    SELECT count(*) INTO unknown_rows
    FROM prediction_logs
    WHERE predicted_trend NOT IN ('BULLISH', 'BEARISH', 'SIDEWAYS');

    RAISE NOTICE
        '048: % wierszy prediction_logs z predicted_trend spoza BULLISH/BEARISH/SIDEWAYS (CHECK NOT VALID ich nie sprawdza)',
        unknown_rows;
END
$migration$;

COMMENT ON COLUMN prediction_logs.predicted_trend IS
    'Kierunek predykcji: BULLISH / BEARISH / SIDEWAYS albo NULL. CHECK prediction_logs_predicted_trend_known (048, NOT VALID) blokuje inne wartości w nowych i zmienianych wierszach.';
