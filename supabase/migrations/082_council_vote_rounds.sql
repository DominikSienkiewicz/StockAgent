-- =====================================================================
-- 082_council_vote_rounds.sql — rundy głosów rady w `council_votes` (ML-40).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: tryb debaty (`debate_mode_enabled`, R-19) robi jedną rundę rewizji, gdy
-- runda 1 rady jest podzielona (BUY i SELL naraz). Do tej pory `council_votes`
-- dostawało tylko głosy po rewizji, więc nie dało się ocenić, czy rewizja poprawia
-- trafność person, czy tylko spłaszcza dysydentów. Od tej migracji zapisujemy obie:
--
--   * `round` — 1 = niezależny głos persony (każdy wiersz sprzed 082 i każdy zapis bez
--     klucza `round`), 2 = głos po rundzie rewizji. CHECK 1–2: debata ma najwyżej
--     jedną rundę rewizji;
--   * indeks UNIQUE `idx_council_votes_prediction_investor_uniq` (054) obejmuje teraz
--     rundę: `(prediction_id, investor_name, round)`. Nazwa zostaje, więc ponowne
--     uruchomienie 054 (`CREATE UNIQUE INDEX IF NOT EXISTS`) nie wraca do starego
--     klucza. Upsert głosów (`council_votes_upsert_enabled`) celuje odtąd w ten klucz
--     i czeka na 082 w `FLAG_REQUIRED_MIGRATION`;
--   * RPC trafności person (015/018/045 w wersji 058, 065) liczą wyłącznie rundę 1:
--     ranking i wagi mierzą niezależny głos persony, jak przed debatą, a runda 2 nie
--     waży drugi raz tej samej predykcji. Ciała bez zmian poza warunkiem `round = 1`;
--     `persona_accuracy_weights` i `persona_accuracy_stats` powtarzają SET search_path
--     z 070 (CREATE OR REPLACE kasuje ustawienia funkcji).
--
-- Bez debaty (flaga domyślnie off) każdy głos to runda 1, więc wyniki RPC i zapis
-- są takie jak przed migracją. Kolumna w istniejącej tabeli — RLS z 038 ją obejmuje.
-- Funkcje nie są nowe: CREATE OR REPLACE zachowuje ACL, a blok niżej powtarza
-- REVOKE / GRANT jak 058 i 065, żeby plik czytał się samodzielnie.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS (CHECK w ADD COLUMN nie dubluje się),
-- indeks przebudowany tylko, gdy nie obejmuje jeszcze rundy, CREATE OR REPLACE
-- FUNCTION, REVOKE / GRANT / COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE council_votes
    ADD COLUMN IF NOT EXISTS round SMALLINT NOT NULL DEFAULT 1
        CONSTRAINT ck_council_votes_round CHECK (round BETWEEN 1 AND 2);

DO $migration$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = 'idx_council_votes_prediction_investor_uniq'
          AND indexdef NOT LIKE '%round%'
    ) THEN
        DROP INDEX public.idx_council_votes_prediction_investor_uniq;
    END IF;
END
$migration$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_council_votes_prediction_investor_uniq
    ON council_votes (prediction_id, investor_name, round);

-- RPC 015: wagi person — ciało z 058, search_path z 070, tylko runda 1 (ML-40).
CREATE OR REPLACE FUNCTION persona_accuracy_weights(window_days INT DEFAULT 90)
RETURNS TABLE (investor_name TEXT, weight DOUBLE PRECISION)
LANGUAGE sql
STABLE
SET search_path = public, extensions, pg_catalog
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
      AND pl.stale_quote IS NOT TRUE
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
      AND cv.round = 1
    GROUP BY cv.investor_name;
$$;

-- RPC 018: ranking person — ciało z 058, search_path z 070, tylko runda 1 (ML-40).
CREATE OR REPLACE FUNCTION persona_accuracy_stats(window_days INT DEFAULT 90)
RETURNS TABLE (
    investor_name TEXT,
    hit_rate DOUBLE PRECISION,
    vote_count BIGINT
)
LANGUAGE sql
STABLE
SET search_path = public, extensions, pg_catalog
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
      AND pl.stale_quote IS NOT TRUE
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
      AND cv.round = 1
    GROUP BY cv.investor_name;
$$;

-- RPC 045: surowe głosy person — ciało z 058, tylko runda 1 (ML-40).
CREATE OR REPLACE FUNCTION persona_vote_outcomes(window_days INT DEFAULT 90)
RETURNS TABLE (
    prediction_id UUID,
    investor_name TEXT,
    recommendation TEXT,
    confidence DOUBLE PRECISION,
    symbol TEXT,
    price_at_prediction NUMERIC,
    actual_price_after_12h NUMERIC,
    predicted_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        cv.prediction_id,
        cv.investor_name::TEXT,
        cv.recommendation::TEXT,
        cv.confidence::DOUBLE PRECISION,
        pl.symbol::TEXT,
        pl.price_at_prediction::NUMERIC,
        pl.actual_price_after_12h::NUMERIC,
        pl.timestamp AS predicted_at,
        pl.settled_at
    FROM council_votes cv
    JOIN prediction_logs pl ON pl.id = cv.prediction_id
    WHERE pl.actual_price_after_12h IS NOT NULL
      AND pl.stale_quote IS NOT TRUE
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
      AND cv.round = 1
    ORDER BY pl.timestamp, cv.prediction_id, cv.investor_name;
$$;

-- RPC 065: Brier person — ciało z 065, tylko runda 1 (ML-40).
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
          AND cv.round = 1
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

REVOKE EXECUTE ON FUNCTION persona_accuracy_weights(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_accuracy_stats(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_vote_outcomes(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_scoring_stats(INT, TEXT[]) FROM PUBLIC;

-- Role API Supabase bez EXECUTE (BEZ-05, jak 038 i 058); service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
    routine text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            FOREACH routine IN ARRAY ARRAY[
                'persona_accuracy_weights(INT)',
                'persona_accuracy_stats(INT)',
                'persona_vote_outcomes(INT)',
                'persona_scoring_stats(INT, TEXT[])'
            ] LOOP
                EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM %I', routine, api_role);
            END LOOP;
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION persona_accuracy_weights(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_accuracy_stats(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_vote_outcomes(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_scoring_stats(INT, TEXT[]) TO service_role;
    END IF;
END
$migration$;

COMMENT ON COLUMN council_votes.round IS
    'ML-40: runda rady — 1 = niezależny głos persony, 2 = głos po jedynej rundzie '
    'rewizji trybu debaty (debate_mode_enabled). Trafność person liczy rundę 1.';
COMMENT ON INDEX idx_council_votes_prediction_investor_uniq IS
    'Jeden głos persony na predykcję i rundę — cel upsertu save_council_votes '
    '(NIEZ-14, od 082 z rundą, ML-40).';
COMMENT ON TABLE council_votes IS
    'Per-investor advisory council votes. One row per investor per prediction and '
    'council round (082: round 1 = independent vote, round 2 = after the debate revision).';
