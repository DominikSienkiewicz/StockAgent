-- migrations/005_council_verdict.sql
-- Adds council_verdict JSONB column to store the advisory council output.
-- Run once against your Supabase project via the SQL editor or CLI.

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS council_verdict JSONB DEFAULT NULL;

COMMENT ON COLUMN prediction_logs.council_verdict IS
    'Advisory council output: 11 investor opinions + consensus verdict (JSON).';
