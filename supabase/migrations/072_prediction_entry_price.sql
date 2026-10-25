-- =====================================================================
-- 072_prediction_entry_price.sql — cena wykonalna od otwarcia sesji (WIA-54).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: Fast Loop chodzi przed sesją (~05:30 UTC), więc `price_at_prediction`
-- to ostatnie zamknięcie. Trafność i paper-PnL liczą od niego także lukę na
-- otwarciu, której czytelnik maila przed sesją nie kupi. Rozliczenie zapisuje
-- obok niego cenę, po której predykcję dało się wykonać:
--
--   * `entry_price` — otwarcie pierwszej sesji NYSE po predykcji, z tego samego
--     notowania /quote (pole `o`), którym reflect rozlicza predykcję; dla
--     predykcji z trwającej sesji — jej własna cena. Mail rysuje z niego drugą
--     krzywą „od otwarcia”.
--
-- NULL = brak: wiersze sprzed tej migracji, krypto (handel 24/7, brak sesji),
-- rozliczenie z zaległości (notowanie z późniejszej sesji), przed sesją po
-- predykcji (weekend, święto) i na nieaktualnym notowaniu (DAT-14), notowanie
-- bez `o` oraz zapis bez flagi. Nigdy zero. Bez backfillu — notowań starych
-- rozliczeń nie ma w bazie.
--
-- Zapisuje Fast Loop za flagą `executable_entry_enabled` (przed migracją klucz →
-- PGRST204 i każdy symbol pada w reflect; preflight FIN-05 zatrzymuje wtedy
-- cykl); odczyt track recordu wybiera kolumnę też tylko za flagą.
-- `price_at_prediction` i `actual_price_after_12h` zostają rozliczeniem
-- (trafność, trening, diagnoza bez zmian). Kolumna w istniejącej tabeli — RLS
-- z 038 ją obejmuje, uprawnienia się nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, CHECK w ADD COLUMN (nie dubluje się
-- przy powtórce), COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE prediction_logs
    -- W skali pozostałych cen (DECIMAL(14,4), ARCH-38); cena jest dodatnia albo jej nie ma.
    ADD COLUMN IF NOT EXISTS entry_price DECIMAL(14, 4) CHECK (entry_price > 0);

COMMENT ON COLUMN prediction_logs.entry_price IS
    'WIA-54: cena wykonalna — otwarcie pierwszej sesji NYSE po predykcji z notowania '
    'rozliczenia (pole o), dla predykcji z trwającej sesji jej cena. NULL = brak '
    '(krypto, zaległe rozliczenie, sprzed 072), nigdy zero.';
