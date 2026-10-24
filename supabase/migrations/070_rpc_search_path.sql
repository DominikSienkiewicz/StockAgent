-- =====================================================================
-- 070_rpc_search_path.sql — stały search_path na RPC aplikacji (BEZ-18).
-- Uruchom PO 059 (i migracjach wcześniejszych), workflow „🗄️ DB Migrate (manual)”.
--
-- Trzy funkcje, które aplikacja woła przez PostgREST, nie miały ustawionego
-- search_path (Supabase advisor `function_search_path_mutable`):
--   * match_news_embeddings(VECTOR, INT, TEXT)  — RAG, 017
--   * persona_accuracy_weights(INT)            — wagi person, 015 / 058
--   * persona_accuracy_stats(INT)              — ranking person, 018 / 058
-- Funkcja bez własnego search_path rozwiązuje nazwy tabel i operatora pgvector `<=>`
-- według ustawienia sesji wołającego, więc obiekt o tej samej nazwie w schemacie
-- wcześniejszym w ścieżce podmieniłby to, co czyta RPC.
--
-- `public` (tabele aplikacji), `extensions` (schemat, do którego Supabase instaluje
-- rozszerzenia z panelu; w czystym Postgresie go nie ma i jest pomijany) i
-- `pg_catalog`. ALTER FUNCTION zmienia tylko konfigurację: ciało, ACL i komentarze
-- zostają, więc kolejna migracja z CREATE OR REPLACE musi sama powtórzyć SET.
--
-- EXECUTE: 038 odebrała je PUBLIC i rolom API na wszystkich funkcjach aplikacji,
-- a 058 powtórzyła to dla RPC person. Blok niżej robi to samo jawnie dla tych trzech,
-- żeby plik czytał się samodzielnie i żeby późniejsze DROP + CREATE (które wracają do
-- domyślnego grantu PUBLIC) nie zostawiły dziury. Role `anon` / `authenticated` /
-- `service_role` istnieją w Supabase, ale nie w gołym kontenerze testowym, więc ich
-- kroki sprawdzają pg_roles (guard); REVOKE FROM PUBLIC działa zawsze.
--
-- Nie obejmuje: persona_vote_outcomes (045), refresh_ml_feature_store (004) i funkcji
-- triggera stamp_prediction_settled_at (029) — poza listą BEZ-18.
--
-- Idempotentne: ALTER FUNCTION ... SET, REVOKE i GRANT można powtarzać.
-- =====================================================================

ALTER FUNCTION match_news_embeddings(VECTOR, INT, TEXT)
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION persona_accuracy_weights(INT)
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION persona_accuracy_stats(INT)
    SET search_path = public, extensions, pg_catalog;

REVOKE EXECUTE ON FUNCTION match_news_embeddings(VECTOR, INT, TEXT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_accuracy_weights(INT) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION persona_accuracy_stats(INT) FROM PUBLIC;

DO $migration$
DECLARE
    api_role text;
    routine text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            FOREACH routine IN ARRAY ARRAY[
                'match_news_embeddings(VECTOR, INT, TEXT)',
                'persona_accuracy_weights(INT)',
                'persona_accuracy_stats(INT)'
            ] LOOP
                EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM %I', routine, api_role);
            END LOOP;
        END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION match_news_embeddings(VECTOR, INT, TEXT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_accuracy_weights(INT) TO service_role;
        GRANT EXECUTE ON FUNCTION persona_accuracy_stats(INT) TO service_role;
    END IF;
END
$migration$;
