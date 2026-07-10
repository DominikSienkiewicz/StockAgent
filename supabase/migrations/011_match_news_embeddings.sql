-- migrations/011_match_news_embeddings.sql
-- RPC similarity search nad prediction_logs.embedding (pgvector) dla RAG.
--
-- Cel: predict_node embeduje podsumowanie newsów bieżącego cyklu i pyta o
-- najbardziej podobne HISTORYCZNE sytuacje wraz z ich wynikiem (czy prognoza
-- się sprawdziła + wniosek), które trafiają do promptu LLM jako analogie.
-- Bez tej funkcji kolumna `embedding` była zapisywana, ale nigdy nieczytana.
--
-- Wymaga: migracji 003 (kolumna embedding VECTOR(1536) + indeks ivfflat
-- z vector_cosine_ops) oraz 009 (is_trend_correct).
--
-- Wywołanie z klienta:
--   supabase.rpc("match_news_embeddings",
--                {"query_embedding": [...1536 floatów...], "match_count": 3})
--
-- Idempotentne (CREATE OR REPLACE).

CREATE OR REPLACE FUNCTION match_news_embeddings(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 3
)
RETURNS TABLE (
    id                  UUID,
    symbol              VARCHAR,
    news_summary        TEXT,
    predicted_trend     VARCHAR,
    is_trend_correct    BOOLEAN,
    correction_insights TEXT,
    similarity          FLOAT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        p.id,
        p.symbol,
        p.news_summary,
        p.predicted_trend,
        p.is_trend_correct,
        p.correction_insights,
        1 - (p.embedding <=> query_embedding) AS similarity   -- cosine similarity
    FROM prediction_logs p
    WHERE p.embedding IS NOT NULL
    ORDER BY p.embedding <=> query_embedding                   -- cosine distance ASC
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_news_embeddings(VECTOR, INT) IS
    'RAG: zwraca match_count historycznych predykcji o najbardziej podobnym '
    'kontekście newsowym (cosine similarity nad embedding). Używane przez '
    'predict_node do wstrzykiwania analogii do promptu.';
