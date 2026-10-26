-- =====================================================================
-- 085_function_search_path_rest.sql — stały search_path na funkcjach z 063, 065 i 073 (BEZ-33).
-- Uruchom PO 084 (i migracjach wcześniejszych), workflow „🗄️ DB Migrate (manual)”.
--
-- 070 (BEZ-18) i 084 (BEZ-17) przypięły search_path sześciu funkcjom; advisor Supabase
-- `function_search_path_mutable` dalej zgłasza pięć kolejnych:
--   * match_news_embeddings_exact(VECTOR, INT, TEXT) — dokładne wyszukiwanie RAG, RPC (063)
--   * persona_scoring_stats(INT, TEXT[])            — Brier i log-loss person, RPC (065 / 082)
--   * paper_trades_chain_head()                     — trigger INSERT `paper_trades` (073)
--   * paper_trades_close_only()                     — trigger UPDATE `paper_trades` (073)
--   * paper_trades_append_only()                    — trigger DELETE / TRUNCATE (073)
-- Bez własnego search_path funkcja rozwiązuje nazwy według ustawienia sesji wołającego:
-- obiekt o tej samej nazwie w schemacie wcześniejszym w ścieżce podmieniłby tabelę,
-- a sesja bez `public` w ścieżce w ogóle jej nie znajdzie.
--
-- Ta sama wartość co w 070 i 084: `public` (tabele aplikacji), `extensions` (schemat
-- rozszerzeń Supabase z operatorem `<=>` pgvector, którego używa wyszukiwanie dokładne;
-- w czystym Postgresie go nie ma i jest pomijany) i `pg_catalog`.
-- ALTER FUNCTION zmienia tylko konfigurację: ciało, właściciel, ACL, komentarze i
-- ustawienia `enable_*` planisty z 063 zostają, więc kolejna migracja z CREATE OR
-- REPLACE tych funkcji musi sama powtórzyć SET (082 zgubiła go przy persona_vote_outcomes).
--
-- Bez SECURITY DEFINER: funkcje dalej działają z prawami wołającego. EXECUTE bez
-- zmian — RPC mają je tylko dla service_role (063, 065, 082), a funkcji triggerów
-- nikt nie woła wprost (073).
--
-- Idempotentne: ALTER FUNCTION ... SET można powtarzać.
-- =====================================================================

ALTER FUNCTION match_news_embeddings_exact(VECTOR, INT, TEXT)
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION persona_scoring_stats(INT, TEXT[])
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION paper_trades_chain_head()
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION paper_trades_close_only()
    SET search_path = public, extensions, pg_catalog;
ALTER FUNCTION paper_trades_append_only()
    SET search_path = public, extensions, pg_catalog;
