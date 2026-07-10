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

-- Podbicie maintenance_work_mem NA CZAS tej migracji. ADD COLUMN ... STORED na
-- prediction_logs przepisuje tabelę i ODBUDOWUJE indeks ivfflat embeddingu
-- (1536-dim, migracja 003), który przy realnych danych potrzebuje ~61 MB.
-- Domyślne 32 MB na mniejszych instancjach Supabase daje błąd 54000
-- ("memory required is 61 MB, maintenance_work_mem is 32 MB"). To USERSET GUC —
-- ustawienie sesyjne, bezpieczne. URUCHOM PLIK JAKO JEDEN BATCH (nie
-- statement-po-statemencie), żeby SET objął ALTER w tej samej sesji.
SET maintenance_work_mem = '256MB';

-- ---------------------------------------------------------------------
-- price_snapshots
-- ---------------------------------------------------------------------
ALTER TABLE price_snapshots
    ADD COLUMN IF NOT EXISTS timestamp_hour TIMESTAMPTZ
        GENERATED ALWAYS AS (date_trunc('hour', timestamp, 'UTC')) STORED;

-- DEDUP PRZED unikalnym indeksem: tabele sprzed tej migracji mają już
-- DUPLIKATY (symbol, timestamp_hour) z ery gołego .insert() (re-run / retry /
-- nakładający się workflow_dispatch). Bez tego CREATE UNIQUE INDEX wywala się
-- błędem 23505 ("Key (symbol, timestamp_hour)=... is duplicated"). Zostawiamy
-- NAJNOWSZY snapshot godziny (po timestamp), resztę usuwamy. Idempotentne —
-- po deduplikacji ponowne uruchomienie nic nie kasuje.
DELETE FROM price_snapshots
WHERE id IN (
    SELECT id FROM (
        SELECT id, row_number() OVER (
            PARTITION BY symbol, timestamp_hour
            ORDER BY timestamp DESC, id DESC
        ) AS rn
        FROM price_snapshots
    ) ranked
    WHERE ranked.rn > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_price_snapshots_symbol_hour_uniq
    ON price_snapshots (symbol, timestamp_hour);

-- ---------------------------------------------------------------------
-- prediction_logs
-- ---------------------------------------------------------------------
ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS timestamp_hour TIMESTAMPTZ
        GENERATED ALWAYS AS (date_trunc('hour', timestamp, 'UTC')) STORED;

-- DEDUP jak wyżej, ale priorytet zachowania ma wiersz ROZLICZONY
-- (is_trend_correct IS NOT NULL) — nie chcemy stracić feedbacku/accuracy przez
-- skasowanie rozliczonej predykcji na rzecz świeższego, nierozliczonego
-- duplikatu. Dopiero potem decyduje świeżość. Skasowanie duplikatu kasuje
-- kaskadowo jego council_votes (FK ON DELETE CASCADE z migracji 007) — głosy
-- zachowanego wiersza zostają.
DELETE FROM prediction_logs
WHERE id IN (
    SELECT id FROM (
        SELECT id, row_number() OVER (
            PARTITION BY symbol, timestamp_hour
            ORDER BY (is_trend_correct IS NOT NULL) DESC, timestamp DESC, id DESC
        ) AS rn
        FROM prediction_logs
    ) ranked
    WHERE ranked.rn > 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_logs_symbol_hour_uniq
    ON prediction_logs (symbol, timestamp_hour);

COMMENT ON COLUMN price_snapshots.timestamp_hour IS
    'date_trunc(hour, timestamp) — klucz idempotencji dla upsertu '
    '(symbol, timestamp_hour). Zapobiega duplikatom przy re-run/retry (#7).';

COMMENT ON COLUMN prediction_logs.timestamp_hour IS
    'date_trunc(hour, timestamp) — klucz idempotencji dla upsertu '
    '(symbol, timestamp_hour). Zapobiega duplikatom przy re-run/retry (#7).';

-- Przywróć domyślny maintenance_work_mem (sprzątanie po podbiciu na górze).
RESET maintenance_work_mem;
