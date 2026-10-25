-- =====================================================================
-- 065_persona_scoring.sql — ocena person regułą Briera i log loss (WIA-44)
-- Uruchom PO 059_settlement_session_close.sql (ręczny workflow „DB Migrate”).
--
-- PO CO: RPC 018 (`persona_accuracy_stats`) liczy AVG(trafienie) i ignoruje
-- pewność głosu: persona mówiąca „BUY 0.95” i myląca się traci tyle samo co
-- „BUY 0.55”, więc hit-rate nagradza przepewne persony (ich wagi z 015 też).
-- Ta funkcja zwraca dla każdej persony: hit-rate, liczbę głosów, średni Brier
-- (confidence − trafienie)² i średni log loss deklarowanej pewności.
-- Aplikacja układa z tego leaderboard po Brier Skill względem stałej prognozy
-- persony (`PersonaTrackRecord.brier_skill`, src/domain/persona_track_record.py),
-- ale tylko za flagą `persona_brier_ranking_enabled`; bez flagi woła dalej
-- 018 / 045 i niczego z tej migracji nie czyta.
--
-- Reguła trafności jak w domenie (`persona_vote_correct`), nie jak w 018:
--   * BUY trafia przy actual > price, SELL przy actual < price;
--   * HOLD trafia, gdy |actual − price| / price mieści się w pasmie „bez ruchu”
--     klasy aktywa: ±2.5% dla symboli z parametru `crypto_symbols`, ±0.5% dla
--     reszty (018 brało ±0.5% dla każdego HOLD — G45, WIA-51);
--   * cena wejścia 0: HOLD trafia tylko przy braku ruchu; rekomendacja spoza
--     BUY/SELL/HOLD to pudło.
-- Log loss przycięty jak w domenie (`log_loss`): prawdopodobieństwo trafienia
-- zawsze w [1e-6, 1 − 1e-6], więc pewność 1.0 przy pudle nie daje nieskończoności.
--
-- Okno `window_days` liczy się jak w 015, 018 i 045: od NOW() po chwili predykcji.
-- Jak po 058 pomijamy rozliczenia z nieaktualnego notowania (`stale_quote IS NOT
-- TRUE`: NULL znaczy „nieznane” i zostaje).
--
-- Uprawnienia jak po 038/045: REVOKE EXECUTE od PUBLIC, GRANT dla service_role
-- w bloku DO z guardem pg_roles (kontener testów nie ma ról Supabase).
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE / GRANT / COMMENT można
-- powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION persona_scoring_stats(
    window_days INT DEFAULT 90,
    crypto_symbols TEXT[] DEFAULT '{}'
)
RETURNS TABLE (
    investor_name TEXT,
    hit_rate DOUBLE PRECISION,
    vote_count BIGINT,
    brier DOUBLE PRECISION,
    log_loss DOUBLE PRECISION
)
LANGUAGE sql
STABLE
AS $$
    WITH scored AS (
        SELECT
            cv.investor_name::TEXT AS investor_name,
            cv.confidence::DOUBLE PRECISION AS confidence,
            (
                CASE
                    WHEN UPPER(BTRIM(cv.recommendation)) = 'BUY'
                         AND pl.actual_price_after_12h > pl.price_at_prediction THEN 1.0
                    WHEN UPPER(BTRIM(cv.recommendation)) = 'SELL'
                         AND pl.actual_price_after_12h < pl.price_at_prediction THEN 1.0
                    WHEN UPPER(BTRIM(cv.recommendation)) = 'HOLD'
                         AND (
                             CASE
                                 WHEN pl.price_at_prediction = 0
                                     THEN pl.actual_price_after_12h = pl.price_at_prediction
                                 ELSE ABS(pl.actual_price_after_12h - pl.price_at_prediction)
                                      / pl.price_at_prediction
                                      <= CASE
                                          WHEN UPPER(pl.symbol) IN (
                                              SELECT UPPER(crypto_symbol)
                                              FROM unnest(crypto_symbols) AS crypto_symbol
                                          ) THEN 0.025
                                          ELSE 0.005
                                      END
                             END
                         ) THEN 1.0
                    ELSE 0.0
                END
            )::DOUBLE PRECISION AS hit
        FROM council_votes cv
        JOIN prediction_logs pl ON pl.id = cv.prediction_id
        WHERE pl.actual_price_after_12h IS NOT NULL
          AND pl.stale_quote IS NOT TRUE
          AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    )
    SELECT
        scored.investor_name,
        AVG(scored.hit)::DOUBLE PRECISION AS hit_rate,
        COUNT(*)::BIGINT AS vote_count,
        AVG(POWER(scored.confidence - scored.hit, 2))::DOUBLE PRECISION AS brier,
        AVG(
            -LN(
                GREATEST(
                    1e-6,
                    LEAST(
                        1.0 - 1e-6,
                        CASE WHEN scored.hit = 1.0 THEN scored.confidence
                             ELSE 1.0 - scored.confidence END
                    )
                )
            )
        )::DOUBLE PRECISION AS log_loss
    FROM scored
    GROUP BY scored.investor_name;
$$;

REVOKE EXECUTE ON FUNCTION persona_scoring_stats(INT, TEXT[]) FROM PUBLIC;

-- Role API Supabase: publiczne bez EXECUTE (BEZ-05, jak 038), service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'REVOKE EXECUTE ON FUNCTION persona_scoring_stats(INT, TEXT[]) FROM %I',
                api_role
            );
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION persona_scoring_stats(INT, TEXT[]) TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION persona_scoring_stats(INT, TEXT[]) IS
    'WIA-44: hit-rate, liczba głosów, średni Brier i log loss deklarowanej pewności '
    'głosu per persona; trafność jak w domenie (HOLD krypto ±2.5% z parametru). '
    'Za flagą persona_brier_ranking_enabled.';
