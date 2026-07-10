-- =====================================================================
-- 022_alpha_fusion_score.sql — ósma cecha ML: fuzja sygnałów alfa (#14)
-- Uruchom w Supabase SQL Editor PO 021_shock_alerts.sql.
--
-- PO CO: 5 źródeł alfa (insiderzy, opcje/put-call, social, itd.) było dotąd
-- pobieranych, klasyfikowanych progami w domenie i renderowanych WYŁĄCZNIE jako
-- tabelka — predykcja LLM i cechy XGBoost nic o nich nie wiedziały. Domena
-- (`alpha_fusion.py`) liczy teraz jeden deterministyczny composite
-- `alpha_fusion_score` (ważona suma klasyfikacji; brakujące źródło = wkład 0).
-- Ta migracja utrwala go na `prediction_logs` i wystawia jako ÓSMĄ cechę
-- treningową w `ml_feature_store`.
--
-- KOREKTA KRYTYCZNA (train/serve-skew): `ml_feature_store` to MATERIALIZED VIEW,
-- więc dołożenie kolumny wymaga DROP + CREATE. Nowa kolumna MUSI wejść jako
-- `COALESCE(alpha_fusion_score, 0.0)`. Powód: kolumna jest dodana późno, więc
-- CAŁA historyczna część danych ma ją NULL. Bez COALESCE `dropna` przy treningu
-- WYCIĄŁBY tę historię (skew: model uczy się tylko na świeżym ogonie). Zerowy
-- default jest neutralny — wszystkie flagi alfa OFF też dają score 0.0, więc
-- historia i „brak sygnału" są nieodróżnialne i spójne.
--
-- Definicja widoku odtworzona 1:1 z migracji 012 (target = ZWROT 12h) i
-- rozszerzona o jedną kolumnę — niczego nie gubiąc.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS + DROP ... IF EXISTS + CREATE.
-- =====================================================================

-- Deterministyczny composite smart-money ∈ (ok. -1..1); NULL dopóki fuzja
-- nie policzyła sygnału dla danej predykcji (stare wiersze, flagi alfa OFF).
ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS alpha_fusion_score DOUBLE PRECISION;

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
        -- Ósma cecha: NULL → 0.0, żeby dropna nie wyciął historii (skew).
        COALESCE(alpha_fusion_score, 0.0) AS alpha_fusion_score,
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
    -- Ósma cecha ML: composite fuzji alfa (już z COALESCE 0.0 z base).
    alpha_fusion_score,
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

COMMENT ON COLUMN prediction_logs.alpha_fusion_score IS
    'Deterministyczny composite fuzji sygnałów alfa (#14); NULL/OFF traktowane jako 0.0 w widoku ML.';
