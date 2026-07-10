-- =====================================================================
-- 012_ml_feature_store_return_target.sql — target = ZWROT 12h
-- Uruchom w Supabase SQL Editor PO 011_match_news_embeddings.sql.
--
-- P0 (ciche błędy poprawności ML):
--   * Target zmieniony z `actual_price_after_12h` (cena BEZWZGLĘDNA) na ZWROT
--     12h: `(actual_price_after_12h - price_current) / price_current
--     AS target_return`. RMSE mierzy teraz trafność RUCHU, nie poziomu ceny
--     (poziom dominował RMSE → model „echujący" ostatnią cenę wygrywał, nie
--     ucząc się niczego). Baseline persystencji to wtedy „zwrot = 0", więc
--     bramka „pobij baseline" w XGBoostAdapter jest sensowna.
--   * `price_delta` BEZ ZMIAN: nadal LAG(price_at_prediction) = zmiana względem
--     POPRZEDNIEJ zalogowanej predykcji. Inference (predict_node) używa tej
--     samej referencji przez RepositoryPort.get_last_prediction_price — to
--     usuwa train/serve skew (wcześniej inference liczył deltę od snapshotu).
--
-- UWAGA WDROŻENIOWA (BREAKING): model `.ubj` wytrenowany na STARYM kontrakcie
-- (cena bezwzględna) jest NIEKOMPATYBILNY z nowym kodem (predict() rekonstruuje
-- cenę z przewidzianego ZWROTU). Po zaaplikowaniu tej migracji USUŃ stary plik
-- modelu (domyślnie `data/models/price_predictor.ubj`). Fast Loop poleci wtedy
-- na cold-start baseline („brak ruchu" = zwrot 0 = bieżąca cena), aż najbliższy
-- Slow Loop wytrenuje model od nowa na nowym targecie.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml_feature_store;

CREATE MATERIALIZED VIEW ml_feature_store AS
WITH base AS (
    SELECT
        id,
        symbol,
        timestamp,
        price_at_prediction AS price_current,
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
        actual_price_after_12h
    FROM prediction_logs
    WHERE actual_price_after_12h IS NOT NULL
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
    -- Target = ZWROT 12h (guard anty-dziel-0 dla skażonych cen bazowych).
    CASE
        WHEN price_current IS NULL OR price_current = 0 THEN NULL
        ELSE (actual_price_after_12h - price_current) / price_current
    END AS target_return
FROM base;

-- Indeksy odtwarzane (DROP MATERIALIZED VIEW usuwa je razem z widokiem).
-- Unikalny indeks po `id` jest wymagany do REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_feature_store_id
    ON ml_feature_store (id);

CREATE INDEX IF NOT EXISTS idx_ml_feature_store_symbol_timestamp
    ON ml_feature_store (symbol, timestamp);
