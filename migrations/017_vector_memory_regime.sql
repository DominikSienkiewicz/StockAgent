-- migrations/017_vector_memory_regime.sql
-- Cross-Asset Vector Memory (regime-tagged RAG).
--
-- Cel: każda predykcja zapisuje, w jakim REŻIMIE rynku powstała. Przy
-- similarity search (RAG) wolimy analogi z tego samego reżimu — bessowy analog
-- mało mówi o ruchu w hossie. Bez tagu retrieval mieszał reżimy i mógł
-- podsuwać mylące precedensy.
--
-- Zmiany:
--   1. Kolumna `regime TEXT` na prediction_logs (NULL = sprzed tagowania).
--   2. Overload RPC match_news_embeddings o opcjonalny `filter_regime`:
--      gdy podany, zwraca analogi z TEGO SAMEGO reżimu LUB nietagowane
--      (regime IS NULL) — żeby pusta jeszcze baza nadal dawała precedensy.
--      Zwraca też kolumnę `regime` (do prezentacji / audytu).
--
-- Wymaga: migracji 011 (oryginalny match_news_embeddings) + 003 (embedding).
-- Idempotentne (ADD COLUMN IF NOT EXISTS, DROP + CREATE OR REPLACE).
-- Backward-compat: feature jest flag-gated (default off); stara 2-argumentowa
-- sygnatura jest zastąpiona przez 3-argumentową z DEFAULT, więc wywołania bez
-- `filter_regime` (klasyczny RAG) działają bez zmian.

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS regime TEXT;

-- Indeks pod filtr reżimu (częściowy — tylko wiersze z tagiem i embeddingiem).
CREATE INDEX IF NOT EXISTS idx_prediction_logs_regime
    ON prediction_logs (regime)
    WHERE regime IS NOT NULL AND embedding IS NOT NULL;

-- Usuwamy starą 2-argumentową wersję, by uniknąć niejednoznaczności overloadu
-- przy rozwiązywaniu wywołania przez PostgREST (nowa wersja ma DEFAULT-y).
DROP FUNCTION IF EXISTS match_news_embeddings(VECTOR, INT);

CREATE OR REPLACE FUNCTION match_news_embeddings(
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
      -- Brak filtra → wszystko. Z filtrem → ten sam reżim albo wiersze
      -- nietagowane (regime IS NULL), by częściowo wypełniona baza działała.
      AND (
          filter_regime IS NULL
          OR p.regime IS NULL
          OR p.regime = filter_regime
      )
    ORDER BY p.embedding <=> query_embedding                   -- cosine distance ASC
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_news_embeddings(VECTOR, INT, TEXT) IS
    'RAG: match_count historycznych predykcji o najbardziej podobnym kontekście '
    'newsowym (cosine similarity nad embedding). filter_regime (opcjonalny) '
    'ogranicza do tego samego reżimu rynku LUB wierszy nietagowanych. Używane '
    'przez predict_node do wstrzykiwania analogii do promptu (regime-tagged).';
