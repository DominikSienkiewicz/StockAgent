-- =====================================================================
-- 040_llm_calls.sql — ledger kosztów: jeden wiersz na request LLM / embeddingów
-- (FIN-13). Uruchom PO 039_feature_store_av_degraded_rows.sql, workflow
-- „🗄️ DB Migrate (manual)”.
--
-- PO CO: sekcja „Koszt cyklu” liczyła koszt jako licznik wywołań × stała
-- (`UNIT_COST_USD`), więc ani długość promptu, ani zmiana modelu, ani cache nie
-- ruszały rachunku. Adaptery zgłaszają teraz tokeny z odpowiedzi dostawcy, domena
-- wycenia je cennikiem per model z `config.toml`, a ta tabela trzyma każdy request
-- do budżetów, raportów i rekoncyliacji z fakturą.
--
-- Kolumny:
--   run_id      — przebieg (`RunContext.run_id`: GITHUB_RUN_ID albo 'local');
--   symbol      — symbol cyklu; NULL dla jobów bez symbolu (Slow Loop);
--   node        — węzeł grafu (reflect, predict, council, save) albo job Slow Loopa;
--   provider, model — dostawca i id modelu, o który prosił adapter (klucz cennika);
--   *_tokens    — kategorie rozłączne: wejście po pełnej stawce, wyjście, odczyt
--                 i zapis cache promptu;
--   cost_usd    — koszt z tokenów; NULL = model spoza cennika (bez zgadywania);
--   price_as_of — data cennika, którym policzono cost_usd.
-- Świadomie BEZ treści promptu i odpowiedzi: prompt niesie kontekst portfela.
--
-- Zapis batchem po przebiegu Fast i Slow Loopa za flagą `cost_ledger_enabled`
-- (przed migracją tabeli nie ma — każdy zapis kończyłby się błędem w logu).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta. Granty dla anon / authenticated
-- odbierają domyślne uprawnienia ustawione w 038.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS llm_calls (
    id                 BIGSERIAL PRIMARY KEY,
    run_id             TEXT NOT NULL,
    symbol             TEXT,
    node               TEXT NOT NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL CONSTRAINT llm_calls_input_tokens_non_negative
                           CHECK (input_tokens >= 0),
    output_tokens      INTEGER NOT NULL CONSTRAINT llm_calls_output_tokens_non_negative
                           CHECK (output_tokens >= 0),
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0
                           CONSTRAINT llm_calls_cache_read_tokens_non_negative
                           CHECK (cache_read_tokens >= 0),
    cache_write_tokens INTEGER NOT NULL DEFAULT 0
                           CONSTRAINT llm_calls_cache_write_tokens_non_negative
                           CHECK (cache_write_tokens >= 0),
    cost_usd           DOUBLE PRECISION CONSTRAINT llm_calls_cost_non_negative
                           CHECK (cost_usd >= 0),
    price_as_of        DATE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Raporty i rekoncyliacja czytają okno czasu („koszt ostatnich 30 dni”).
CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at
    ON llm_calls (created_at DESC);

ALTER TABLE llm_calls ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE llm_calls IS
    'Ledger kosztów: jeden request LLM / embeddingów (tokeny, koszt, przebieg), bez treści promptu (FIN-13).';
COMMENT ON COLUMN llm_calls.cost_usd IS
    'Koszt z tokenów po cenniku z price_as_of; NULL = model spoza cennika.';
COMMENT ON COLUMN llm_calls.input_tokens IS
    'Wejście po pełnej stawce — bez tokenów odczytu i zapisu cache (osobne kolumny).';
