-- =====================================================================
-- 071_schema_version.sql — `public.schema_version()`: do której migracji sięga baza
-- (NIEZ-30). Uruchom PO 070, workflow „🗄️ DB Migrate (manual)”.
--
-- Flaga zależna od migracji, włączona przed jej zaaplikowaniem, kończyła się PGRST204
-- (zapis nieistniejącej kolumny) i gubiła cały zapis predykcji; do tej pory
-- ostrzegały przed tym tylko komentarze w config.toml. Entry-pointy pytają teraz o
-- najwyższą zaaplikowaną wersję i na czas przebiegu wyłączają flagi, których migracji
-- baza jeszcze nie ma (mapa `FLAG_REQUIRED_MIGRATION` w src/domain/feature_requirements.py).
--
-- Funkcja czyta rejestr Supabase CLI (`supabase_migrations.schema_migrations`),
-- do którego klucz secret (service_role) nie ma dostępu, więc jest SECURITY DEFINER
-- (właściciel to rola migracji) i zwraca wyłącznie liczbę. `SET search_path = ''` i
-- nazwy z kwalifikatorem schematu: SECURITY DEFINER z ruchomą ścieżką pozwalałby
-- podstawić obiekt.
--
-- Zwraca NULL, gdy wersji nie da się ustalić: rejestru nie ma (baza spoza CLI) albo
-- jest pusty (migracje wklejane ręcznie do SQL Editora). NULL to „nie wiem”, nigdy 0 —
-- aplikacja zostawia wtedy flagi włączone i loguje WARNING (decyzja właściciela,
-- fail-open). Wersje z samych cyfr (001, 070) są numerami z `supabase/migrations/`;
-- wersje w formacie znacznika czasu (14 cyfr, `supabase migration new`) pomijamy.
--
-- EXECUTE: tylko service_role. Role `anon` / `authenticated` / `service_role` istnieją
-- w Supabase, ale nie w gołym kontenerze testowym, więc ich kroki sprawdzają pg_roles
-- (guard); REVOKE FROM PUBLIC działa zawsze.
--
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE, GRANT i COMMENT można powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION public.schema_version()
RETURNS INT
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    latest INT;
BEGIN
    SELECT max(version::INT)
    INTO latest
    FROM supabase_migrations.schema_migrations
    WHERE version ~ '^[0-9]{1,6}$';
    RETURN latest;
EXCEPTION
    WHEN undefined_table OR invalid_schema_name THEN
        RETURN NULL;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.schema_version() FROM PUBLIC;

DO $migration$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE EXECUTE ON FUNCTION public.schema_version() FROM anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE EXECUTE ON FUNCTION public.schema_version() FROM authenticated;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION public.schema_version() TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION public.schema_version() IS
    'Najwyższy numer migracji z supabase/migrations/ zapisany w rejestrze Supabase CLI '
    '(supabase_migrations.schema_migrations); NULL, gdy rejestru nie ma albo jest pusty. '
    'Aplikacja wyłącza na przebieg flagi, których migracji baza jeszcze nie ma (NIEZ-30).';
