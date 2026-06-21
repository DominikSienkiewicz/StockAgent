-- =====================================================================
-- 015_persona_accuracy.sql — RPC wag person rady wg trafności (#3)
-- Uruchom w Supabase SQL Editor PO 014_subscribers.sql.
--
-- Adaptive persona weighting: głos persony w konsensusie rady jest ważony jej
-- HISTORYCZNĄ trafnością — czy jej rekomendacja (BUY/SELL/HOLD) zgadzała się z
-- realnym ruchem ceny rozliczonej predykcji. RPC liczy to w SQL (Python tylko
-- mapuje wynik). Waga ∈ [0.5, 1.5]: 0.5+accuracy, clamp.
--   - BUY trafny gdy actual > price_at_prediction
--   - SELL trafny gdy actual < price_at_prediction
--   - HOLD trafny gdy |actual-price|/price <= 0.005 (±0.5%)
-- Tylko predykcje rozliczone (actual_price_after_12h IS NOT NULL) z okna.
-- =====================================================================

CREATE OR REPLACE FUNCTION persona_accuracy_weights(window_days INT DEFAULT 90)
RETURNS TABLE (investor_name TEXT, weight DOUBLE PRECISION)
LANGUAGE sql
STABLE
AS $$
    SELECT
        cv.investor_name,
        GREATEST(0.5, LEAST(1.5, 0.5 + AVG(
            CASE
                WHEN cv.recommendation = 'BUY'
                     AND pl.actual_price_after_12h > pl.price_at_prediction THEN 1.0
                WHEN cv.recommendation = 'SELL'
                     AND pl.actual_price_after_12h < pl.price_at_prediction THEN 1.0
                WHEN cv.recommendation = 'HOLD'
                     AND pl.price_at_prediction <> 0
                     AND ABS(pl.actual_price_after_12h - pl.price_at_prediction)
                         / pl.price_at_prediction <= 0.005 THEN 1.0
                ELSE 0.0
            END
        )))::DOUBLE PRECISION AS weight
    FROM council_votes cv
    JOIN prediction_logs pl ON pl.id = cv.prediction_id
    WHERE pl.actual_price_after_12h IS NOT NULL
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    GROUP BY cv.investor_name;
$$;
