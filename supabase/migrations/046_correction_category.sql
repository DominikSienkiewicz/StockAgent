-- =====================================================================
-- 046_correction_category.sql — kategoria przyczyny błędu predykcji (WIA-57)
-- Uruchom PO 045_persona_vote_outcomes.sql (ręczny workflow „DB Migrate”).
--
-- PO CO: panel lessons pokazywał „powracający motyw” jako najczęstsze słowo
-- diagnoz, czyli dla realistycznych diagnoz słowo w rodzaju „zignorowałem”,
-- a nie przyczynę. Diagnoza LLM zwraca teraz JSON {category, insight}, gdzie
-- kategoria pochodzi z zamkniętego zbioru `MistakeCategory`
-- (src/domain/mistake_taxonomy.py). Tekst diagnozy dalej trafia do
-- `correction_insights`, a kategoria do tej kolumny — raport liczy z niej
-- histogram („3 z 5 błędów: makro”).
--
-- NULL = brak kategorii: wiersz sprzed migracji, predykcja trafiona albo
-- placeholder bez płatnej diagnozy. Bez backfillu — kategorii starych diagnoz
-- nie da się odtworzyć bez nowego wywołania LLM.
--
-- Kod zapisuje i czyta kolumnę dopiero za flagą `correction_category_enabled`
-- — przed migracją nowy klucz zwróciłby PGRST204 i zgubił rozliczenie.
--
-- Nowa kategoria w domenie = nowa migracja zmieniająca CHECK.
-- Idempotentne: ADD COLUMN IF NOT EXISTS (z CHECK), COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS correction_category TEXT
        CHECK (correction_category IN (
            'MACRO', 'EARNINGS', 'STALE_NEWS', 'SECTOR_ROTATION',
            'SENTIMENT_TRAP', 'NOISE', 'OTHER'
        ));

COMMENT ON COLUMN prediction_logs.correction_category IS
    'WIA-57: kategoria przyczyny błędu z diagnozy LLM (MistakeCategory). '
    'NULL = brak diagnozy z kategorią (trafiona, placeholder, wiersz sprzed 046).';
