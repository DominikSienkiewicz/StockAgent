-- =====================================================================
-- 078_diagnosis_judge.sql — ocena diagnoz sędzią z rubryką (EXP-10).
-- Uruchom PO 077, workflow „🗄️ DB Migrate (manual)”.
--
-- Slow Loop (flaga `diagnosis_judge_enabled`) co tydzień ocenia losową próbkę
-- najwyżej 10 diagnoz błędnych predykcji rubryką trzech kryteriów tak/nie:
-- czynnik obecny w newsach lub cenach, teza falsyfikowalna, nie ogólnik.
--   diagnosis_judge_score — liczba spełnionych kryteriów (0–3); NULL = bez oceny,
--                           czyli kandydat do kolejnej próbki.
--   rejected_diagnosis    — tekst diagnozy z wynikiem poniżej progu; jej miejsce
--                           w `correction_insights` zajmuje placeholder, który
--                           lessons, analogi RAG i recap odsiewają. Tekst zostaje
--                           tutaj, więc odrzucenie da się cofnąć.
--
-- Bez nowej tabeli i funkcji (RLS i granty z 038 obejmują kolumny tabeli).
-- Idempotentne: ADD COLUMN IF NOT EXISTS (z ograniczeniem) i COMMENT.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS diagnosis_judge_score SMALLINT
        CHECK (diagnosis_judge_score BETWEEN 0 AND 3),
    ADD COLUMN IF NOT EXISTS rejected_diagnosis TEXT;

COMMENT ON COLUMN prediction_logs.diagnosis_judge_score IS
    'EXP-10: ocena diagnozy sędzią z rubryką — ile z 3 kryteriów (czynnik w dowodach, '
    'falsyfikowalność, konkret) spełnia; NULL = bez oceny.';
COMMENT ON COLUMN prediction_logs.rejected_diagnosis IS
    'EXP-10: diagnoza odrzucona przez sędziego (wynik poniżej progu) — poza lessons i RAG.';
