-- migrations/008_data_quality_flags.sql
-- Adds data_quality_flags column to prediction_logs.
--
-- Cel: rozróżnić "feature = 0.0 bo realnie zero" od "feature = 0.0 bo
-- AlphaVantage zwrócił None/NaN". Bez tej kolumny model trenuje na skażonych
-- danych bez śladu. Lista flag postaci ["av_sentiment_score_missing",
-- "news_volume_24h_invalid"] — przy treningu filtruj
-- WHERE jsonb_array_length(data_quality_flags) = 0.
--
-- Idempotentne (IF NOT EXISTS).

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS data_quality_flags JSONB DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_prediction_logs_data_quality_clean
    ON prediction_logs ((jsonb_array_length(data_quality_flags)))
    WHERE data_quality_flags IS NOT NULL;

COMMENT ON COLUMN prediction_logs.data_quality_flags IS
    'List of input quality flags detected during this prediction cycle. Empty array = all inputs clean.';
