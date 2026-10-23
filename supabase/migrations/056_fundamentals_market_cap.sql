-- =====================================================================
-- 056_fundamentals_market_cap.sql — kapitalizacja spółki w cache
-- fundamentów (DAT-21).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: próg znaczącego zakupu insidera (kod P officera z Form 4) był stały,
-- 100k USD — w spółce za 4 bln USD to szum, w spółce za 2 mld USD wyraźny
-- sygnał. DAT-21 skaluje próg kapitalizacją (ułamek 1e-6). Slow Loop bierze ją
-- z `MarketCapitalization` tej samej odpowiedzi Alpha Vantage OVERVIEW, którą
-- już pobiera (zero nowych requestów), a Fast Loop czyta ją wyłącznie z cache.
--
-- Wartość w USD jak w OVERVIEW; NULL = brak w odpowiedzi albo wpis sprzed tej
-- migracji (próg wraca wtedy do 100k USD). Bez backfillu — kolejny refresh
-- Slow Loopa (TTL cache 6 dni) uzupełnia wpisy sam.
--
-- Zapis za flagą `fundamentals_market_cap_enabled` (przed migracją nowy klucz
-- w upsercie → PGRST204 i utrata całego wpisu cache). Odczyt `select("*")`
-- działa z kolumną i bez niej. Kolumna w istniejącej tabeli — RLS z 038 ją
-- obejmuje, uprawnienia się nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE fundamentals_cache
    -- DAT-21: kapitalizacja w USD z OVERVIEW; NULL = nieznana.
    ADD COLUMN IF NOT EXISTS market_cap NUMERIC;

COMMENT ON COLUMN fundamentals_cache.market_cap IS
    'Kapitalizacja w USD z Alpha Vantage OVERVIEW (MarketCapitalization, DAT-21) — skaluje próg znaczącego zakupu insidera. NULL = nieznana.';
