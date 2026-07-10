-- =====================================================================
-- 020_decision_receipts.sql — persystowany audit trail predykcji (#13)
-- Uruchom w Supabase SQL Editor PO 019_model_scorecards.sql.
--
-- Kwity decyzyjne: dziś analogi RAG, atrybucje SHAP-lite i odznaki proweniencji
-- są render-only (`agent_graph.py`: "precedent receipts dla raportu, brak
-- persystencji") — znikają po wysłaniu maila. Gdy predykcja po 12h okazuje się
-- błędna, nie da się zaudytować, na jakich danych stała. Ta kolumna JSONB
-- zapisuje pełny łańcuch dowodowy każdej predykcji (precedensy, atrybucje,
-- próg efektywny), więc post-mortem jest możliwy.
--
-- MUSI być NULLABLE: stare wiersze (sprzed migracji) oraz predykcje bez kwitów
-- renderują się jak dziś (brak kolumny = brak sekcji). Bez nullowalności stary
-- kod insertujący bez tej kolumny by się wywalił.
--
-- schema_version: payload JSONB niesie własne pole `schema_version` (np. 1),
-- żeby czytnik bezpiecznie rozpoznawał starsze wersje schematu kwitu i migrował
-- odczyt bez psucia historii (dryf schematu JSONB pod kontrolą wersji).
--
-- Idempotentne (ADD COLUMN IF NOT EXISTS).
-- =====================================================================

ALTER TABLE prediction_logs
    -- JSONB, nullable. Payload wersjonowany polem `schema_version` w środku.
    ADD COLUMN IF NOT EXISTS decision_receipts JSONB;

COMMENT ON COLUMN prediction_logs.decision_receipts IS
    'Audit trail predykcji (precedensy, atrybucje SHAP-lite, próg efektywny). '
    'Nullable. Payload niesie schema_version do bezpiecznego odczytu starych wersji.';
