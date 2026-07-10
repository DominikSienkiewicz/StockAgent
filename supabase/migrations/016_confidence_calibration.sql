-- =====================================================================
-- 016_confidence_calibration.sql — ocena kalibracji pewności (#4)
-- Uruchom w Supabase SQL Editor PO 015_persona_accuracy.sql.
--
-- LLM-as-Judge calibration: po rozliczeniu predykcji Slow-Loop'owy sędzia-LLM
-- ocenia, czy DEKLAROWANA pewność była uzasadniona tym, co było wiadome.
--   confidence_calibration ∈ [0,1] — 1 = pewność trafnie skalibrowana.
--   calibration_insight    — krótkie uzasadnienie sędziego.
-- Oba nullable (wypełniane dopiero w Slow Loop, gdy `calibration_judge_enabled`).
-- =====================================================================

ALTER TABLE prediction_logs
    -- Deklarowana pewność predykcji (z LLM) — wcześniej nie persystowana, a bez
    -- niej nie da się ocenić kalibracji. Nullable (stare wiersze = NULL).
    ADD COLUMN IF NOT EXISTS confidence_score       FLOAT,
    ADD COLUMN IF NOT EXISTS confidence_calibration FLOAT,
    ADD COLUMN IF NOT EXISTS calibration_insight    TEXT;
