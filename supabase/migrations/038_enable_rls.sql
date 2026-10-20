-- =====================================================================
-- 038_enable_rls.sql — Row Level Security na każdej tabeli `public` i
-- odebranie uprawnień rolom API `anon` / `authenticated` (BEZ-05).
-- Uruchom PO 037_experiment_arms.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PROSTUJE uzasadnienie z 001_init.sql i 002_price_snapshots.sql („projekt
-- backend-only, RLS tylko by przeszkadzał”). Tamto założenie było błędne:
-- Supabase wystawia schemat `public` przez PostgREST każdemu, kto zna URL
-- projektu i klucz PUBLICZNY (anon / publishable), a domyślne uprawnienia
-- Supabase dają rolom `anon` i `authenticated` pełne prawa do każdej nowej
-- tabeli, widoku, sekwencji i funkcji w `public`. Bez RLS e-maile
-- subskrybentów (`subscribers`) i portfel właściciela (`positions`) chroniła
-- wyłącznie tajność klucza, który z definicji nie jest tajny.
-- RLS nie przeszkadza aplikacji: łączy się kluczem secret / service_role,
-- a rola `service_role` ma BYPASSRLS. Właściciel tabel (rola migracji,
-- `postgres`) też omija RLS, bo nie ustawiamy FORCE ROW LEVEL SECURITY.
--
-- Co robi:
--  1. ENABLE ROW LEVEL SECURITY BEZ polityk na każdej tabeli `public`
--     (relkind r/p, bez obiektów rozszerzeń). Lista jest wyliczana z pg_class,
--     więc obejmuje też tabele dodane ręcznie poza migracjami. Brak polityk =
--     rola bez BYPASSRLS nie zobaczy ani nie zapisze żadnego wiersza.
--  2. REVOKE od `anon` i `authenticated`: ALL na tabelach (ALL TABLES obejmuje
--     też widoki i widok zmaterializowany `ml_feature_store`, który RLS nie
--     ma), ALL na sekwencjach, EXECUTE na funkcjach.
--  3. ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ... od tych samych ról
--     dla tabel, sekwencji i funkcji tworzonych przez rolę migracji. Inaczej
--     DROP + CREATE widoku (wzorzec 022/026/027) przywróciłby granty z
--     domyślnych uprawnień Supabase.
--  4. REVOKE EXECUTE ... FROM PUBLIC na funkcjach aplikacji (bez funkcji
--     rozszerzeń, np. pgvector): Postgres domyślnie daje EXECUTE roli PUBLIC,
--     a `anon` jest jej członkiem, więc samo REVOKE od `anon` nie wystarcza.
--     `service_role` dostaje EXECUTE jawnie — aplikacja woła RPC
--     (`match_news_embeddings`, `persona_accuracy_weights`,
--     `persona_accuracy_stats`, `refresh_ml_feature_store`) tym kluczem.
--     Trigger 029 (`stamp_prediction_settled_at`) działa dalej: Postgres
--     sprawdza EXECUTE funkcji triggera tylko przy CREATE TRIGGER.
--
-- Role `anon` / `authenticated` / `service_role` istnieją w Supabase, ale nie
-- w gołym kontenerze testowym — kroki 2–4 sprawdzają pg_roles (guard), a
-- krok 1 i REVOKE FROM PUBLIC działają zawsze.
--
-- Pułapki:
--  - 001 i 002 jawnie robią DISABLE ROW LEVEL SECURITY. Ręczne ponowne
--    uruchomienie 001/002 (SQL Editor) wyłączy RLS na `prediction_logs` /
--    `price_snapshots` — po nim trzeba ponownie uruchomić tę migrację.
--  - Nowa tabela w kolejnej migracji potrzebuje własnego ENABLE ROW LEVEL
--    SECURITY, a nowa funkcja REVOKE EXECUTE ... FROM PUBLIC (i GRANT EXECUTE
--    TO service_role, jeśli woła ją aplikacja). Pilnują tego testy
--    kontenerowe w tests/infrastructure/test_migrations.py.
--
-- Kolejność wdrożenia (docs/data-and-access.md): najpierw klucz secret /
-- service_role w GitHub Secrets i lokalnym .env, potem deploy kodu, dopiero
-- na końcu ta migracja — klucz publiczny po niej nie odczyta niczego.
--
-- Idempotentne: ENABLE RLS, REVOKE i GRANT można powtarzać bez skutków ubocznych.
-- =====================================================================

-- 1. RLS bez polityk na każdej tabeli `public` (poza obiektami rozszerzeń).
DO $migration$
DECLARE
    target regclass;
BEGIN
    FOR target IN
        SELECT c.oid::regclass
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND NOT c.relrowsecurity
          AND NOT EXISTS (
              SELECT 1
              FROM pg_depend d
              WHERE d.classid = 'pg_class'::regclass
                AND d.objid = c.oid
                AND d.deptype = 'e'
          )
    LOOP
        EXECUTE format('ALTER TABLE %s ENABLE ROW LEVEL SECURITY', target);
    END LOOP;
END
$migration$;

-- 2–3. Role API Supabase: bez uprawnień do istniejących i przyszłych obiektów.
DO $migration$
DECLARE
    api_role text;
BEGIN
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
            EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', api_role);
            EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', api_role);
            EXECUTE format(
                'REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM %I', api_role
            );
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM %I',
                api_role
            );
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON SEQUENCES FROM %I',
                api_role
            );
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM %I',
                api_role
            );
        END IF;
    END LOOP;
END
$migration$;

-- 4. Funkcje aplikacji: bez domyślnego EXECUTE dla PUBLIC, jawnie dla service_role.
DO $migration$
DECLARE
    routine regprocedure;
    grant_service_role boolean;
BEGIN
    grant_service_role := EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role');
    FOR routine IN
        SELECT p.oid::regprocedure
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND p.prokind = 'f'
          AND NOT EXISTS (
              SELECT 1
              FROM pg_depend d
              WHERE d.classid = 'pg_proc'::regclass
                AND d.objid = p.oid
                AND d.deptype = 'e'
          )
    LOOP
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC', routine);
        IF grant_service_role THEN
            EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', routine);
        END IF;
    END LOOP;
END
$migration$;
