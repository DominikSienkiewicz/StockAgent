-- =====================================================================
-- 073_paper_trades.sql — append-only paper ledger z łańcuchem hashy (WIA-59).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: paper-PnL track recordu jest dziś przeliczany z mutowalnych kolumn
-- `prediction_logs` — `update_prediction_accuracy` nadpisuje rozliczenie, a ręczny
-- re-run upsertem nadpisuje predykcję z tej samej godziny. Ledger zapisuje pozycję
-- papierową RAZ, przy zapisie predykcji (save), i zamyka ją RAZ, przy rozliczeniu
-- (reflect); krzywa kapitału może czytać go zamiast `prediction_logs`.
--
-- Łańcuch: `row_hash = sha256(canonicalize(wejście pozycji) + prev_hash)` liczy
-- aplikacja (domena `src/domain/paper_ledger.py`), a pierwszy wiersz ma
-- `prev_hash` = 64 zera. Baza pilnuje, że łańcuch jest liniowy:
--
--   * UNIQUE(prev_hash) — dwa zapisy z tą samą głową (symbol_concurrency > 1)
--     nie rozgałęzią łańcucha; przegrany dostaje 23505 i ponawia od nowej głowy;
--   * trigger BEFORE INSERT — `prev_hash` musi być bieżącą głową (`row_hash`
--     wiersza o największym `seq`, a dla pustej tabeli 64 zera), inaczej 40001
--     (serialization_failure), a nowy wiersz wchodzi otwarty (bez zamknięcia);
--   * trigger BEFORE UPDATE — jedyny dozwolony UPDATE to zamknięcie otwartej
--     pozycji (exit_price, exit_ts, pnl, exit_reason naraz); zamknięta pozycja
--     i kolumny wejścia są niezmienne;
--   * trigger BEFORE DELETE i BEFORE TRUNCATE — ledger jest append-only;
--     `service_role` nie ma też uprawnień DELETE ani TRUNCATE.
--
-- Hash obejmuje wejście (decyzję). Zamknięcie to cena rynkowa sprawdzalna
-- z zewnątrz, a trigger pozwala zapisać je tylko raz. Głowę łańcucha Fast Loop
-- publikuje raz dziennie obok atestacji (`paper_ledger_heads.jsonl`).
--
-- Zapisuje i czyta Fast Loop za flagą `paper_ledger_enabled` (przed migracją
-- tabeli nie ma). Nowa tabela = RLS bez polityk (jak po 038); funkcje triggerów
-- bez EXECUTE dla PUBLIC (Postgres sprawdza je tylko przy CREATE TRIGGER).
--
-- Idempotentne: CREATE TABLE / INDEX IF NOT EXISTS, CREATE OR REPLACE FUNCTION,
-- DROP TRIGGER IF EXISTS + CREATE TRIGGER, REVOKE / COMMENT można powtarzać.
-- =====================================================================

CREATE TABLE IF NOT EXISTS paper_trades (
    seq            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id  UUID NOT NULL
                       CONSTRAINT paper_trades_prediction_unique UNIQUE
                       REFERENCES prediction_logs (id),
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL CONSTRAINT paper_trades_side_known
                       CHECK (side IN ('long', 'short', 'flat')),
    entry_price    DECIMAL(14, 4) NOT NULL CONSTRAINT paper_trades_entry_positive
                       CHECK (entry_price > 0),
    entry_ts       TIMESTAMPTZ NOT NULL,
    weight         NUMERIC(8, 4) NOT NULL CONSTRAINT paper_trades_weight_positive
                       CHECK (weight > 0),
    cost_bps       NUMERIC(8, 2) NOT NULL CONSTRAINT paper_trades_cost_non_negative
                       CHECK (cost_bps >= 0),
    prev_hash      TEXT NOT NULL
                       CONSTRAINT paper_trades_prev_hash_unique UNIQUE
                       CONSTRAINT paper_trades_prev_hash_hex CHECK (prev_hash ~ '^[0-9a-f]{64}$'),
    row_hash       TEXT NOT NULL
                       CONSTRAINT paper_trades_row_hash_unique UNIQUE
                       CONSTRAINT paper_trades_row_hash_hex CHECK (row_hash ~ '^[0-9a-f]{64}$'),
    exit_price     DECIMAL(14, 4),
    exit_ts        TIMESTAMPTZ,
    pnl            NUMERIC(14, 8),
    exit_reason    TEXT CONSTRAINT paper_trades_exit_reason_known
                       CHECK (exit_reason IN ('settled', 'stale_quote')),
    recorded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Zamknięcie zapisuje się w całości albo wcale.
    CONSTRAINT paper_trades_close_complete CHECK (
        (exit_price IS NULL AND exit_ts IS NULL AND pnl IS NULL AND exit_reason IS NULL)
        OR (exit_price IS NOT NULL AND exit_ts IS NOT NULL AND pnl IS NOT NULL
            AND exit_reason IS NOT NULL)
    )
);

-- Krzywa kapitału czyta okno po chwili wejścia.
CREATE INDEX IF NOT EXISTS idx_paper_trades_entry_ts ON paper_trades (entry_ts DESC);

ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION paper_trades_chain_head()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    head TEXT;
BEGIN
    SELECT row_hash INTO head FROM paper_trades ORDER BY seq DESC LIMIT 1;
    IF NEW.prev_hash IS DISTINCT FROM COALESCE(head, repeat('0', 64)) THEN
        RAISE EXCEPTION 'paper_trades: prev_hash % is not the chain head %',
            NEW.prev_hash, COALESCE(head, 'genesis')
            USING ERRCODE = 'serialization_failure';
    END IF;
    IF NEW.exit_price IS NOT NULL OR NEW.exit_reason IS NOT NULL THEN
        RAISE EXCEPTION 'paper_trades: a trade opens without its close';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION paper_trades_close_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.exit_price IS NOT NULL THEN
        RAISE EXCEPTION 'paper_trades: trade % is closed and immutable', OLD.seq;
    END IF;
    IF NEW.exit_price IS NULL THEN
        RAISE EXCEPTION 'paper_trades: an update of trade % must close it', OLD.seq;
    END IF;
    IF (NEW.seq, NEW.prediction_id, NEW.symbol, NEW.side, NEW.entry_price, NEW.entry_ts,
        NEW.weight, NEW.cost_bps, NEW.prev_hash, NEW.row_hash, NEW.recorded_at)
        IS DISTINCT FROM
       (OLD.seq, OLD.prediction_id, OLD.symbol, OLD.side, OLD.entry_price, OLD.entry_ts,
        OLD.weight, OLD.cost_bps, OLD.prev_hash, OLD.row_hash, OLD.recorded_at) THEN
        RAISE EXCEPTION 'paper_trades: only the close of trade % may be written', OLD.seq;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION paper_trades_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'paper_trades is append-only: % is not allowed', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_paper_trades_chain_head ON paper_trades;
CREATE TRIGGER trg_paper_trades_chain_head
    BEFORE INSERT ON paper_trades
    FOR EACH ROW
    EXECUTE FUNCTION paper_trades_chain_head();

DROP TRIGGER IF EXISTS trg_paper_trades_close_only ON paper_trades;
CREATE TRIGGER trg_paper_trades_close_only
    BEFORE UPDATE ON paper_trades
    FOR EACH ROW
    EXECUTE FUNCTION paper_trades_close_only();

DROP TRIGGER IF EXISTS trg_paper_trades_no_delete ON paper_trades;
CREATE TRIGGER trg_paper_trades_no_delete
    BEFORE DELETE ON paper_trades
    FOR EACH ROW
    EXECUTE FUNCTION paper_trades_append_only();

DROP TRIGGER IF EXISTS trg_paper_trades_no_truncate ON paper_trades;
CREATE TRIGGER trg_paper_trades_no_truncate
    BEFORE TRUNCATE ON paper_trades
    FOR EACH STATEMENT
    EXECUTE FUNCTION paper_trades_append_only();

REVOKE EXECUTE ON FUNCTION paper_trades_chain_head() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION paper_trades_close_only() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION paper_trades_append_only() FROM PUBLIC;

-- Role API Supabase: publiczne bez EXECUTE (BEZ-05, jak 038); `service_role` pisze
-- i zamyka pozycje, ale nie usuwa ich ani nie czyści tabeli (druga warstwa obok
-- triggerów). Funkcji triggerów nikt nie woła wprost, więc bez GRANT EXECUTE.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'REVOKE EXECUTE ON FUNCTION paper_trades_chain_head(), '
                'paper_trades_close_only(), paper_trades_append_only() FROM %I',
                api_role
            );
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        REVOKE DELETE, TRUNCATE ON paper_trades FROM service_role;
    END IF;
END
$migration$;

COMMENT ON TABLE paper_trades IS
    'WIA-59: append-only paper ledger — pozycja otwarta przy zapisie predykcji, zamknięta raz '
    'przy rozliczeniu; row_hash = sha256(canonicalize(wejście) + prev_hash), łańcuch liniowy.';

COMMENT ON COLUMN paper_trades.row_hash IS
    'WIA-59: sha256 kanonicznego wejścia pozycji (src/domain/paper_ledger.py, ledger_version 1) '
    'sklejonego z prev_hash; głowa łańcucha = row_hash wiersza o największym seq.';

COMMENT ON COLUMN paper_trades.pnl IS
    'WIA-59: zwrot brutto pozycji (ułamek) — sygnał strony × zmiana ceny od wejścia; koszt '
    'obrotu liczy krzywa portfela.';

COMMENT ON COLUMN paper_trades.exit_reason IS
    'WIA-59: settled = rozliczenie predykcji; stale_quote = rozliczenie na notowaniu sprzed '
    'sesji po predykcji (DAT-14) — pozycja zamknięta, ale poza krzywą kapitału.';
