-- =====================================================================
-- 063_rag_exact_search.sql — dokładne wyszukiwanie analogów RAG (ML-23).
-- Uruchom workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: indeks ivfflat z migracji 003 ma lists = 100, a tabela kilka tysięcy
-- wierszy z embeddingiem. Przy domyślnym `ivfflat.probes = 1` zapytanie przeszukuje
-- ~1% list, więc „najbardziej podobne analogi” bywają przypadkowe, a z filtrem
-- reżimu (017) wynik potrafi mieć mniej niż `match_count` wierszy — RAG i rerank
-- pracują na wejściu, którego nikt nie mierzy.
--
-- Analogi trafiają do promptu predykcji, a od 2026-10-26 do 2027-01-29 trwa EXP-03
-- (A/B pamięci). Dlatego ta migracja NIE rusza indeksu ani `match_news_embeddings`:
-- dodaje osobną funkcję `match_news_embeddings_exact` o tym samym kontrakcie, którą
-- aplikacja woła dopiero za flagą `rag_exact_search_enabled`.
--
-- Funkcja wyłącza na czas wywołania skany indeksowe i jawnie włącza skan
-- sekwencyjny (SET w definicji funkcji, niezależnie od ustawień sesji), więc planer
-- robi skan sekwencyjny z sortowaniem top-N — dokładne KNN. Przy < 50 tys.
-- wierszy to milisekundy i nie wymaga budowy HNSW (maintenance_work_mem na planie
-- Free) ani CREATE INDEX CONCURRENTLY poza transakcją.
--
-- Uprawnienia jak po 038: REVOKE EXECUTE od PUBLIC, GRANT dla service_role w bloku
-- DO z guardem pg_roles (kontener testów nie ma ról Supabase).
--
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE / GRANT / COMMENT można
-- powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION match_news_embeddings_exact(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 3,
    filter_regime TEXT DEFAULT NULL
)
RETURNS TABLE (
    id                  UUID,
    symbol              VARCHAR,
    news_summary        TEXT,
    predicted_trend     VARCHAR,
    is_trend_correct    BOOLEAN,
    correction_insights TEXT,
    regime              TEXT,
    similarity          FLOAT
)
LANGUAGE sql
STABLE
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
        1 - (p.embedding <=> query_embedding) AS similarity   -- cosine similarity
    FROM prediction_logs p
    WHERE p.embedding IS NOT NULL
      -- Ten sam filtr co `match_news_embeddings` (017): ten sam reżim albo wiersze
      -- nietagowane.
      AND (
          filter_regime IS NULL
          OR p.regime IS NULL
          OR p.regime = filter_regime
      )
    ORDER BY p.embedding <=> query_embedding                   -- cosine distance ASC
    LIMIT match_count;
$$;

REVOKE EXECUTE ON FUNCTION match_news_embeddings_exact(VECTOR, INT, TEXT) FROM PUBLIC;

-- Role API Supabase: publiczne bez EXECUTE (BEZ-05, jak 038), service_role woła RPC.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format(
                'REVOKE EXECUTE ON FUNCTION match_news_embeddings_exact(VECTOR, INT, TEXT) FROM %I',
                api_role
            );
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION match_news_embeddings_exact(VECTOR, INT, TEXT)
            TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION match_news_embeddings_exact(VECTOR, INT, TEXT) IS
    'ML-23: kontrakt match_news_embeddings, ale dokładne KNN (skan sekwencyjny, '
    'bez ivfflat z probes = 1) — pełne match_count wierszy także z filtrem reżimu. '
    'Aplikacja woła ją za flagą rag_exact_search_enabled.';
