-- =====================================================================
-- 013_idempotency_and_pagination.sql — idempotentne zapisy (#7)
-- Uruchom w Supabase SQL Editor PO 012_ml_feature_store_return_target.sql.
-- Idempotentne (ADD COLUMN IF NOT EXISTS / CREATE UNIQUE INDEX IF NOT EXISTS).
--
-- Problem (#7): save_price_snapshot i save_prediction robiły zwykły .insert()
-- bez klucza idempotencji. Re-run GHA, nakładający się workflow_dispatch albo
-- retry LangGraph wstawiał DUPLIKAT wiersza. Duplikaty podwójnie liczyły się
-- w get_accuracy_stats oraz w zmaterializowanym widoku ml_feature_store,
-- zaburzając raportowaną trafność.
--
-- Rozwiązanie: generowana kolumna timestamp_hour = date_trunc('hour',
-- timestamp, 'UTC') + UNIQUE INDEX na (symbol, timestamp_hour). Adapter
-- przełącza się na .upsert(..., on_conflict="symbol,timestamp_hour"), więc
-- ponowny zapis w TEJ SAMEJ godzinie aktualizuje istniejący logiczny wiersz
-- zamiast tworzyć duplikat. Mirroruje intencję NULL-guardu z
-- update_prediction_accuracy.
--
-- Uwaga: kolumna timestamptz wymaga 3-argumentowego date_trunc z jawną strefą
-- ('UTC') — wariant 2-argumentowy zależy od sesyjnego TimeZone i nie jest
-- IMMUTABLE, więc Postgres odrzuciłby go w generated column.
-- =====================================================================

-- ---------------------------------------------------------------------
-- price_snapshots
-- ---------------------------------------------------------------------
ALTER TABLE price_snapshots
    ADD COLUMN IF NOT EXISTS timestamp_hour TIMESTAMPTZ
        GENERATED ALWAYS AS (date_trunc('hour', timestamp, 'UTC')) STORED;

CREATE UNIQUE INDEX IF NOT EXISTS idx_price_snapshots_symbol_hour_uniq
    ON price_snapshots (symbol, timestamp_hour);

-- ---------------------------------------------------------------------
-- prediction_logs
-- ---------------------------------------------------------------------
ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS timestamp_hour TIMESTAMPTZ
        GENERATED ALWAYS AS (date_trunc('hour', timestamp, 'UTC')) STORED;

CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_logs_symbol_hour_uniq
    ON prediction_logs (symbol, timestamp_hour);

COMMENT ON COLUMN price_snapshots.timestamp_hour IS
    'date_trunc(hour, timestamp) — klucz idempotencji dla upsertu '
    '(symbol, timestamp_hour). Zapobiega duplikatom przy re-run/retry (#7).';

COMMENT ON COLUMN prediction_logs.timestamp_hour IS
    'date_trunc(hour, timestamp) — klucz idempotencji dla upsertu '
    '(symbol, timestamp_hour). Zapobiega duplikatom przy re-run/retry (#7).';
