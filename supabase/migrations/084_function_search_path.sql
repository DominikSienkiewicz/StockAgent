-- =====================================================================
-- 084_function_search_path.sql — stały search_path na pozostałych funkcjach (BEZ-17).
-- Uruchom PO 083 (i migracjach wcześniejszych), workflow „🗄️ DB Migrate (manual)”.
--
-- 070 (BEZ-18) przypięła search_path trzem RPC i świadomie pominęła te trzy funkcje:
--   * refresh_ml_feature_store()      — REFRESH widoku cech, Slow Loop i backtest (001 / 004)
--   * persona_vote_outcomes(INT)      — surowe głosy person, RPC (045 / 058 / 082)
--   * stamp_prediction_settled_at()   — funkcja triggera `prediction_logs` (029)
-- Bez własnego search_path funkcja rozwiązuje nazwy według ustawienia sesji wołającego:
-- obiekt o tej samej nazwie w schemacie wcześniejszym w ścieżce podmieniłby widok albo
-- tabelę, a sesja bez `public` w ścieżce w ogóle ich nie znajdzie.
--
-- Ta sama wartość co w 070: `public` (tabele aplikacji), `extensions` (schemat
-- rozszerzeń Supabase; w czystym Postgresie go nie ma i jest pomijany) i `pg_catalog`.
-- ALTER FUNCTION zmienia tylko konfigurację: ciało, właściciel, ACL i komentarze
-- zostają, więc kolejna migracja z CREATE OR REPLACE musi sama powtórzyć SET
-- (082 przedefiniowała persona_vote_outcomes bez SET — stąd ta funkcja tutaj).
--
-- Bez SECURITY DEFINER: funkcje dalej działają z prawami wołającego. EXECUTE bez
-- zmian — 038 odebrała je PUBLIC i rolom API na wszystkich funkcjach aplikacji
-- (service_role ma je jawnie), a 082 powtórzyła to dla persona_vote_outcomes.
--
-- Idempotentne: ALTER FUNCTION ... SET można powtarzać.
-- =====================================================================

ALTER FUNCTION refresh_ml_feature_store()
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION persona_vote_outcomes(INT)
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION stamp_prediction_settled_at()
    SET search_path = public, extensions, pg_catalog;
