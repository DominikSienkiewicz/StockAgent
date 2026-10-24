-- =====================================================================
-- 058_settled_reads_without_stale_quote.sql — rozliczenia z nieaktualnego
-- notowania poza danymi treningowymi i poza trafnością person (DAT-14).
-- Uruchom po migracjach o niższych numerach, w tym 055 (`stale_quote`)
-- i 050 (ręczny workflow „DB Migrate").
--
-- PO CO: `stale_quote = true` (055) znaczy, że rozliczenie użyło notowania
-- sprzed pierwszej sesji po predykcji (weekend, święto NYSE, drugi przebieg
-- przed otwarciem) — cena rozliczenia równa się cenie wejścia. Odczyty
-- trafności w aplikacji odsiewają takie wiersze od 055, ale trzy ścieżki
-- czytają rozliczenia w SQL i nadal je liczyły:
--   * widok `ml_feature_store` — etykieta 0 % trafiała do treningu XGBoost;
--   * RPC 015 `persona_accuracy_weights` i 018 `persona_accuracy_stats` —
--     wagi i ranking person bez flagi `persona_vote_outcomes_enabled`;
--   * RPC 045 `persona_vote_outcomes` — te same głosy za tą flagą.
-- Filtr to `stale_quote IS NOT TRUE`, nie `= false`: NULL (krypto, wiersze
-- sprzed 055, zapis bez flagi `stale_quote_enabled`) znaczy „nieznane” i zostaje.
--
-- Widok odtworzony 1:1 z 050 (LAG po WSZYSTKICH zalogowanych predykcjach —
-- także stęchłych, bo inference bierze referencję tak samo; filtry rozliczenia
-- i jakości PO LAG; COALESCE-y; `settled_at`, `label_horizon_hours`). Nowy jest
-- tylko warunek `stale_quote IS NOT TRUE` w zewnętrznym WHERE. DROP MATERIALIZED
-- VIEW kasuje indeksy i komentarze — odtwarzamy oba indeksy (unikalny po `id`
-- jest wymagany do REFRESH ... CONCURRENTLY) i komentarze z 050.
--
-- RPC: CREATE OR REPLACE z tą samą sygnaturą i typem wyniku — ciało 015, 018
-- i 045 bez zmian poza warunkiem w WHERE. Uprawnienia jak po 038/045: REVOKE
-- EXECUTE od PUBLIC, GRANT dla service_role w bloku DO z guardem pg_roles.
--
-- Aplikacja niczego nie przełącza: RPC wołane są bez listy kolumn, a trening
-- czyta widok `select("*")`, więc przed 058 działa jak dotąd.
--
-- Idempotentne: DROP ... IF EXISTS + CREATE + CREATE INDEX IF NOT EXISTS,
-- CREATE OR REPLACE FUNCTION; REVOKE, GRANT i COMMENT można powtarzać.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml_feature_store;

CREATE MATERIALIZED VIEW ml_feature_store AS
WITH base AS (
    SELECT
        id,
        symbol,
        timestamp,
        price_at_prediction AS price_current,
        -- LAG po WSZYSTKICH zalogowanych predykcjach symbolu (ML-17, 047).
        LAG(price_at_prediction, 1) OVER (
            PARTITION BY symbol ORDER BY timestamp
        ) AS price_prev_12h,
        LAG(price_at_prediction, 2) OVER (
            PARTITION BY symbol ORDER BY timestamp
        ) AS price_prev_24h,
        sentiment_score,
        COALESCE(av_relevance_avg, 0.0) AS av_relevance_avg,
        COALESCE(news_volume_24h, 0) AS news_volume_24h,
        COALESCE(high_relevance_count, 0) AS high_relevance_count,
        CASE
            WHEN predicted_trend = 'BULLISH'  THEN  1
            WHEN predicted_trend = 'BEARISH'  THEN -1
            ELSE 0
        END AS llm_trend_signal,
        COALESCE(av_llm_agreement, 0.5) AS av_llm_agreement,
        -- Ósma cecha: NULL → 0.0, żeby dropna nie wyciął historii (skew).
        COALESCE(alpha_fusion_score, 0.0) AS alpha_fusion_score,
        -- Wersja zestawu cech; NULL zostaje NULL (wersja nieznana).
        feature_set_version,
        -- Flagi jakości tylko do filtra w zewnętrznym SELECT (po LAG).
        COALESCE(data_quality_flags, '[]'::jsonb) AS data_quality_flags,
        actual_price_after_12h,
        -- ML-25: chwila rozliczenia (029) — horyzont etykiety liczy zewnętrzny SELECT.
        settled_at,
        -- DAT-14: znacznik nieaktualnego notowania (055) tylko do filtra po LAG.
        stale_quote
    FROM prediction_logs
)
SELECT
    id,
    symbol,
    timestamp,
    price_current,
    price_prev_12h,
    price_prev_24h,
    -- price_delta = zmiana względem POPRZEDNIEJ predykcji (guard anty-dziel-0).
    CASE
        WHEN price_prev_12h IS NULL OR price_prev_12h = 0 THEN NULL
        ELSE (price_current - price_prev_12h) / price_prev_12h
    END AS price_delta,
    sentiment_score,
    sentiment_score AS av_sentiment_score,
    av_relevance_avg,
    news_volume_24h,
    high_relevance_count,
    llm_trend_signal,
    av_llm_agreement,
    -- Ósma cecha ML: composite fuzji alfa (już z COALESCE 0.0 z base).
    alpha_fusion_score,
    -- Filtr treningu po bieżącej wersji robi `get_feature_store_data` (flaga).
    feature_set_version,
    -- Target = ZWROT 12h (guard anty-dziel-0 dla skażonych cen bazowych).
    CASE
        WHEN price_current IS NULL OR price_current = 0 THEN NULL
        ELSE (actual_price_after_12h - price_current) / price_current
    END AS target_return,
    settled_at,
    -- Horyzont etykiety w godzinach; dolna granica 029 (settled_at = timestamp)
    -- to brak pomiaru, nie 0 h.
    (
        CASE
            WHEN settled_at IS NULL OR settled_at <= timestamp THEN NULL
            ELSE EXTRACT(EPOCH FROM (settled_at - timestamp)) / 3600.0
        END
    )::double precision AS label_horizon_hours
FROM base
-- Tylko rozliczone predykcje mają etykietę (ML-17: filtr PO LAG). Atrapa LLM
-- i każdy niepełny feed AV to nie dane treningowe (ML-12, EXP-07, DAT-07, DAT-08).
-- DAT-14: rozliczenie z notowania sprzed sesji po predykcji też nie (NULL zostaje).
WHERE actual_price_after_12h IS NOT NULL
  AND stale_quote IS NOT TRUE
  AND NOT (
    data_quality_flags ?| ARRAY[
        'llm_analysis_failed',
        'av_keys_exhausted',
        'av_unexpected_payload',
        'av_fetch_error'
    ]
);

-- Indeksy odtwarzane (DROP MATERIALIZED VIEW usuwa je razem z widokiem).
-- Unikalny indeks po `id` jest wymagany do REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);

-- RPC 015: wagi person — ciało bez zmian poza warunkiem DAT-14.
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
      AND pl.stale_quote IS NOT TRUE
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    GROUP BY cv.investor_name;
$$;

-- RPC 018: ranking person — ciało bez zmian poza warunkiem DAT-14.
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
      AND pl.stale_quote IS NOT TRUE
      AND pl.timestamp >= NOW() - (window_days || ' days')::INTERVAL
    GROUP BY cv.investor_name;
$$;

-- RPC 045: surowe głosy person — ciało bez zmian poza warunkiem DAT-14.
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
    ORDER BY pl.timestamp, cv.prediction_id, cv.investor_name;
$$;

REVOKE EXECUTE ON FUNCTION persona_accuracy_weights(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_accuracy_stats(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_vote_outcomes(INT) FROM PUBLIC;

-- Role API Supabase bez uprawnień do widoku i RPC (BEZ-05, jak 038);
-- service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
    routine text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format('REVOKE ALL ON ml_feature_store FROM %I', api_role);
            FOREACH routine IN ARRAY ARRAY[
                'persona_accuracy_weights(INT)',
                'persona_accuracy_stats(INT)',
                'persona_vote_outcomes(INT)'
            ] LOOP
                EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM %I', routine, api_role);
            END LOOP;
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION persona_accuracy_weights(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_accuracy_stats(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_vote_outcomes(INT) TO service_role;
    END IF;
END
$migration$;

COMMENT ON MATERIALIZED VIEW ml_feature_store IS
    'Cechy treningowe XGBoost: rozliczone predykcje bez flag llm_analysis_failed, av_keys_exhausted (026), av_unexpected_payload, av_fetch_error (039) i bez rozliczeń z nieaktualnego notowania (stale_quote = true, 058). LAG po wszystkich zalogowanych predykcjach, przed filtrami (047). feature_set_version z 027; settled_at i label_horizon_hours z 050.';

COMMENT ON COLUMN ml_feature_store.price_prev_12h IS
    'Cena POPRZEDNIEJ zalogowanej predykcji symbolu (LAG 1 po całej prediction_logs, także nierozliczonych i oflagowanych) — ta sama referencja co get_last_prediction_price w inference. Nazwa historyczna: odstęp to odstęp kolejnych predykcji, nie 12 h.';

COMMENT ON COLUMN ml_feature_store.price_prev_24h IS
    'Cena predykcji sprzed DWÓCH zalogowanych predykcji symbolu (LAG 2 po całej prediction_logs). Nazwa historyczna: odstęp to dwie kolejne predykcje, nie 24 h. Nie jest cechą modelu.';

COMMENT ON COLUMN ml_feature_store.settled_at IS
    'Chwila rozliczenia predykcji (prediction_logs.settled_at, 029). Dla wierszy sprzed triggera przybliżenie z price_snapshots albo dolna granica = timestamp.';

COMMENT ON COLUMN ml_feature_store.label_horizon_hours IS
    'Prawdziwy horyzont etykiety: (settled_at - timestamp) w godzinach (ML-25). NULL = brak pomiaru (settled_at NULL albo dolna granica 029).';

COMMENT ON FUNCTION persona_vote_outcomes(INT) IS
    'WIA-51: rozliczone głosy person bez oceny — trafność liczy domena '
    '(pasmo HOLD wg klasy aktywa). Oś czasu predicted_at / settled_at dla wag '
    'prequential (WIA-46). DAT-14 (058): bez rozliczeń ze stale_quote = true.';

COMMENT ON FUNCTION persona_accuracy_weights(INT) IS
    'Ścieżka sprzed WIA-51: HOLD zawsze ±0.5%, także na krypto. Za flagą '
    'persona_vote_outcomes_enabled aplikacja liczy wagi z persona_vote_outcomes. '
    'DAT-14 (058): bez rozliczeń ze stale_quote = true.';

COMMENT ON FUNCTION persona_accuracy_stats(INT) IS
    'Ścieżka sprzed WIA-51: HOLD zawsze ±0.5%, także na krypto. Za flagą '
    'persona_vote_outcomes_enabled aplikacja liczy ranking z persona_vote_outcomes. '
    'DAT-14 (058): bez rozliczeń ze stale_quote = true.';
