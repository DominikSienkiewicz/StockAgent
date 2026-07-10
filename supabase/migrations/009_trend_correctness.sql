-- migrations/009_trend_correctness.sql
-- Adds is_trend_correct column to prediction_logs.
--
-- Cel: rozdzielić poprawność KIERUNKOWĄ predykcji od liczbowego
-- accuracy_score. accuracy_score mierzy bliskość ceny do predicted_target_price
-- — przy cold-starcie (XGBoost niewytrenowany) target = cena bieżąca, więc
-- każda mało zmienna spółka dostaje ~100% niezależnie od kierunku. Raport
-- pokazywał obok etykiety trendu tę liczbę i twierdził, że błędna kierunkowo
-- prognoza była "trafna". is_trend_correct (BULLISH→wzrost, BEARISH→spadek,
-- SIDEWAYS→ruch w ±0.5%) napędza teraz trafność raportu i get_accuracy_stats.
--
-- Idempotentne (IF NOT EXISTS + backfill tylko dla NULL).

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS is_trend_correct BOOLEAN;

-- Backfill już zamkniętych predykcji — bez tego 30-dniowa "Historia trafności"
-- byłaby pusta do miesiąca po wdrożeniu. Logika odwzorowuje
-- domain.Prediction.is_trend_correct (SIDEWAYS_TOLERANCE = 0.005).
UPDATE prediction_logs
SET is_trend_correct = CASE
        WHEN predicted_trend = 'BULLISH'
            THEN actual_price_after_12h > price_at_prediction
        WHEN predicted_trend = 'BEARISH'
            THEN actual_price_after_12h < price_at_prediction
        WHEN predicted_trend = 'SIDEWAYS'
            THEN abs(
                (actual_price_after_12h - price_at_prediction) / price_at_prediction
            ) <= 0.005
        ELSE NULL
    END
WHERE is_trend_correct IS NULL
  AND actual_price_after_12h IS NOT NULL
  AND price_at_prediction IS NOT NULL
  AND price_at_prediction <> 0;

COMMENT ON COLUMN prediction_logs.is_trend_correct IS
    'Whether the predicted_trend direction matched the realised price move. '
    'NULL until the prediction is resolved by the reflect node. Drives report '
    'hit-rate — accuracy_score (price proximity) does not capture direction.';
