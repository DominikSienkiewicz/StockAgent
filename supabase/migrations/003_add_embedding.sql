-- =====================================================================
-- 003_add_embedding.sql — kolumna wektora embeddingu w prediction_logs
-- Uruchom w Supabase SQL Editor (Project → SQL Editor → New query).
-- Idempotentne (IF NOT EXISTS).
-- =====================================================================

-- pgvector — wymagane do typu VECTOR
CREATE EXTENSION IF NOT EXISTS vector;

-- Kolumna opcjonalna (NULL gdy embedding wyłączony lub API niedostępne)
ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS embedding VECTOR(1536);

-- Indeks IVFFlat do wyszukiwania nearest-neighbor (RAG / podobne predykcje)
CREATE INDEX IF NOT EXISTS idx_prediction_logs_embedding
    ON prediction_logs
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
