-- =====================================================================
-- 076_rag_point_in_time.sql — analogi RAG point-in-time (ML-41).
-- Uruchom PO 075, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: `match_news_embeddings` (011/017) i `match_news_embeddings_exact` (063)
-- zwracają każdy wiersz z embeddingiem — także predykcje nierozliczone, zapisane
-- chwilę wcześniej w TYM cyklu (echo bieżących newsów) i, przy replayu albo
-- backteście, wiersze z przyszłości względem odtwarzanej chwili. Analog bez wyniku
-- zajmuje miejsce w top-N i jest „neutralny” w reranku (rag_rerank: None → 0).
--
-- Nowa funkcja `match_news_embeddings_pit` (kontrakt tamtych plus trzy parametry):
--   * tylko analogi ROZLICZONE (`is_trend_correct IS NOT NULL`);
--   * `before_ts` — tylko predykcje utworzone i rozliczone przed tą chwilą
--     (`settled_at` z 029; NULL w `settled_at` liczy się jako `timestamp`, dolna
--     granica jak w backfillu 029). NULL = bez odcięcia w czasie;
--   * `exclude_symbol` + `exclude_window_hours` — pomija predykcje tego samego
--     symbolu z okna horyzontu przed `before_ts` (albo przed now(), gdy NULL):
--     to echo tych samych newsów, nie historia;
--   * `timestamp` w wyniku — wołający widzi, z kiedy jest analog.
-- Dokładne KNN jak w 063 (skan sekwencyjny wymuszony w definicji funkcji): po
-- filtrach ivfflat z probes = 1 oddawałby jeszcze mniej wierszy niż `match_count`.
--
-- Analogi trafiają do promptu predykcji, a od 2026-10-26 do 2027-01-29 trwa EXP-03
-- (A/B pamięci). Dlatego migracja NIE rusza istniejących funkcji: aplikacja woła
-- nową dopiero za flagą `rag_point_in_time_enabled` (RECIPE, domyślnie off; bramka
-- schematu NIEZ-30 wyłącza ją, dopóki baza nie ma 076).
--
-- Uprawnienia jak po 038: REVOKE EXECUTE od PUBLIC, GRANT dla service_role w bloku
-- DO z guardem pg_roles (kontener testów nie ma ról Supabase). Stały search_path jak
-- po 070 (BEZ-18).
--
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE / GRANT / COMMENT można
-- powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION match_news_embeddings_pit(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 3,
    filter_regime TEXT DEFAULT NULL,
    before_ts TIMESTAMPTZ DEFAULT NULL,
    exclude_symbol TEXT DEFAULT NULL,
    exclude_window_hours INT DEFAULT 72
)
RETURNS TABLE (
    id                  UUID,
    symbol              VARCHAR,
    news_summary        TEXT,
    predicted_trend     VARCHAR,
    is_trend_correct    BOOLEAN,
    correction_insights TEXT,
    regime              TEXT,
    "timestamp"         TIMESTAMPTZ,
    similarity          FLOAT
)
LANGUAGE sql
STABLE
SET search_path = public, extensions, pg_catalog
SET enable_seqscan = on
SET enable_indexscan = off
SET enable_bitmapscan = off
AS $$
    SELECT
        p.id,
        p.symbol,
        p.news_summary,
        p.predicted_trend,
        p.is_trend_correct,
        p.correction_insights,
        p.regime,
        p."timestamp",
        1 - (p.embedding <=> query_embedding) AS similarity   -- cosine similarity
    FROM prediction_logs p
    WHERE p.embedding IS NOT NULL
      -- Tylko analogi z wynikiem: nierozliczony nie ma czego nauczyć.
      AND p.is_trend_correct IS NOT NULL
      -- Ten sam filtr reżimu co 017/063: ten sam reżim albo wiersze nietagowane.
      AND (
          filter_regime IS NULL
          OR p.regime IS NULL
          OR p.regime = filter_regime
      )
      -- Point-in-time: predykcja i jej rozliczenie przed odtwarzaną chwilą.
      AND (
          before_ts IS NULL
          OR (
              p."timestamp" < before_ts
              AND COALESCE(p.settled_at, p."timestamp") < before_ts
          )
      )
      -- Ten sam symbol z okna horyzontu to echo bieżących newsów.
      AND NOT (
          exclude_symbol IS NOT NULL
          AND p.symbol = exclude_symbol
          AND p."timestamp"
              >= COALESCE(before_ts, now()) - make_interval(hours => exclude_window_hours)
      )
    ORDER BY p.embedding <=> query_embedding                   -- cosine distance ASC
    LIMIT match_count;
$$;

REVOKE EXECUTE ON FUNCTION match_news_embeddings_pit(VECTOR, INT, TEXT, TIMESTAMPTZ, TEXT, INT)
    FROM PUBLIC;

-- Role API Supabase: publiczne bez EXECUTE (BEZ-05, jak 038), service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'REVOKE EXECUTE ON FUNCTION '
                'match_news_embeddings_pit(VECTOR, INT, TEXT, TIMESTAMPTZ, TEXT, INT) FROM %I',
                api_role
            );
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION
            match_news_embeddings_pit(VECTOR, INT, TEXT, TIMESTAMPTZ, TEXT, INT)
            TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION match_news_embeddings_pit(VECTOR, INT, TEXT, TIMESTAMPTZ, TEXT, INT) IS
    'ML-41: analogi RAG point-in-time — tylko rozliczone przed before_ts, bez '
    'exclude_symbol z okna exclude_window_hours, z timestamp w wyniku; dokładne KNN '
    'jak 063. Aplikacja woła ją za flagą rag_point_in_time_enabled.';
