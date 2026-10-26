-- =====================================================================
-- 077_slow_loop_rpcs.sql — RPC Slow Loopa: zbiorczy zapis ocen kalibracji (WIA-65)
-- i raport rozmiaru bazy (NIEZ-37).
-- Uruchom PO 076, workflow „🗄️ DB Migrate (manual)”.
--
-- save_calibration_scores(p_scores, p_score_version) — WIA-65 (decyzja 2026-10-23,
-- opcja b): kalibracja pewności zostaje arytmetyczna (bez sędziego LLM), a N
-- aktualizacji `prediction_logs` wiersz po wierszu zastępuje JEDNO wywołanie.
-- `p_scores` to tablica JSON `[{"id": uuid, "score": float}, …]`; funkcja ustawia
-- `confidence_calibration` (016) i `calibration_score_version` (044) wskazanym
-- wierszom i zwraca liczbę zaktualizowanych. `calibration_insight` zostaje bez zmian
-- (wniosek LLM już nie powstaje). Wiersz spoza tablicy i nieznane id nic nie zmieniają.
--
-- SECURITY INVOKER (domyślne): woła ją klucz secret (service_role, BYPASSRLS), który
-- i tak ma UPDATE na tabeli — funkcja nie poszerza uprawnień. `SET search_path = ''`
-- i nazwy z kwalifikatorem schematu (BEZ-18).
--
-- db_size_report() — NIEZ-37 (decyzja 2026-10-24: tylko raport i WARNING, bez
-- retencji): rozmiar każdej relacji schematu `public` (tabela, widok
-- zmaterializowany; z indeksami i TOAST) z `pg_total_relation_size`, od największej.
-- Slow Loop sumuje ją i ostrzega powyżej 70% `db_size_budget_mb` — darmowy plan
-- Supabase po przekroczeniu limitu przechodzi w tryb read-only i zapisy cicho padają.
-- SECURITY DEFINER (właściciel to rola migracji, jak 071): czyta wyłącznie katalog
-- systemowy i zwraca nazwę relacji z liczbą bajtów — żadnych wierszy danych.
-- `SET search_path = ''` i nazwy z kwalifikatorem schematu: SECURITY DEFINER
-- z ruchomą ścieżką pozwalałby podstawić obiekt.
--
-- EXECUTE (obie funkcje): tylko service_role. Role `anon` / `authenticated` /
-- `service_role` istnieją w Supabase, ale nie w gołym kontenerze testowym, więc ich
-- kroki sprawdzają pg_roles (guard); REVOKE FROM PUBLIC działa zawsze.
--
-- Idempotentne: CREATE OR REPLACE FUNCTION, REVOKE, GRANT i COMMENT można powtarzać.
-- =====================================================================

CREATE OR REPLACE FUNCTION public.save_calibration_scores(
    p_scores JSONB,
    p_score_version SMALLINT
)
RETURNS INT
LANGUAGE plpgsql
VOLATILE
SET search_path = ''
AS $$
DECLARE
    updated INT;
BEGIN
    UPDATE public.prediction_logs AS p
    SET confidence_calibration = s.score,
        calibration_score_version = p_score_version
    FROM jsonb_to_recordset(p_scores) AS s(id UUID, score DOUBLE PRECISION)
    WHERE p.id = s.id;
    GET DIAGNOSTICS updated = ROW_COUNT;
    RETURN updated;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.save_calibration_scores(JSONB, SMALLINT) FROM PUBLIC;

DO $migration$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE EXECUTE ON FUNCTION public.save_calibration_scores(JSONB, SMALLINT) FROM anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE EXECUTE ON FUNCTION public.save_calibration_scores(JSONB, SMALLINT)
            FROM authenticated;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION public.save_calibration_scores(JSONB, SMALLINT)
            TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION public.save_calibration_scores(JSONB, SMALLINT) IS
    'WIA-65: arytmetyczne oceny kalibracji pewności jednym zapisem — '
    '[{id, score}] → confidence_calibration i calibration_score_version (1 liniowa, '
    '2 Brier); zwraca liczbę zaktualizowanych wierszy.';

CREATE OR REPLACE FUNCTION public.db_size_report()
RETURNS TABLE (relation_name TEXT, total_bytes BIGINT)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT c.relname::TEXT, pg_catalog.pg_total_relation_size(c.oid)
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p', 'm')
    ORDER BY 2 DESC, 1;
$$;

REVOKE EXECUTE ON FUNCTION public.db_size_report() FROM PUBLIC;

DO $migration$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE EXECUTE ON FUNCTION public.db_size_report() FROM anon;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE EXECUTE ON FUNCTION public.db_size_report() FROM authenticated;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT EXECUTE ON FUNCTION public.db_size_report() TO service_role;
    END IF;
END
$migration$;

COMMENT ON FUNCTION public.db_size_report() IS
    'NIEZ-37: rozmiar relacji schematu public (pg_total_relation_size: z indeksami i '
    'TOAST), od największej; Slow Loop ostrzega powyżej 70% db_size_budget_mb.';
