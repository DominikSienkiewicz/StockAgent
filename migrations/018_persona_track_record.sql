-- =====================================================================
-- 018_persona_track_record.sql — RPC track recordu person rady (#3)
-- Uruchom w Supabase SQL Editor PO 017_vector_memory_regime.sql.
--
-- Persona leaderboard: raport pokazuje, KTO Z RADY MIAŁ RACJĘ — "Buffett: 68%
-- trafności (22 głosy) / Wood: 41% (18)". Ten sam JOIN i te same reguły
-- trafności co `persona_accuracy_weights` (migracja 015), ale zwracany jest
-- SUROWY hit-rate ∈ [0, 1] razem z COUNT(*) rozliczonych głosów.
--
-- Dlaczego osobne RPC, a nie reuse 015: tamto zwraca już PRZESKALOWANĄ wagę
-- konsensusu (GREATEST(0.5, LEAST(1.5, 0.5 + accuracy))), z której nie da się
-- odtworzyć ani czystej trafności, ani wielkości próbki. Bez liczby głosów
-- leaderboard nie odróżni persony 100%/1 głos od persony 68%/22 głosy — a to
-- jest dokładnie ten szum, który próg `min_votes` w domenie ma odciąć.
--
--   - BUY trafny gdy actual > price_at_prediction
--   - SELL trafny gdy actual < price_at_prediction
--   - HOLD trafny gdy |actual-price|/price <= 0.005 (±0.5%)
-- Tylko predykcje rozliczone (actual_price_after_12h IS NOT NULL) z okna.
-- =====================================================================

CREATE OR REPLACE FUNCTION persona_accuracy_stats(window_days INT DEFAULT 90)
RETURNS TABLE (
    investor_name TEXT,
    hit_rate DOUBLE PRECISION,
    vote_count BIGINT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        cv.investor_name,
        AVG(
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
        )::DOUBLE PRECISION AS hit_rate,
        COUNT(*)::BIGINT AS vote_count
    FROM council_votes cv
    JOIN prediction_logs pl ON pl.id = cv.prediction_id
    WHERE pl.actual_price_after_12h IS NOT NULL
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    GROUP BY cv.investor_name;
$$;
