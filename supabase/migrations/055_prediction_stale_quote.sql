-- =====================================================================
-- 055_prediction_stale_quote.sql — znacznik nieaktualnego notowania przy
-- predykcji i rozliczeniu (DAT-14).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: Finnhub /quote w weekend, w święto NYSE i przed otwarciem oddaje
-- zamknięcie poprzedniej sesji. Reflect rozliczał wtedy predykcję ceną równą
-- cenie wejścia (BULLISH i BEARISH = pudło, SIDEWAYS = trafienie), a czasu
-- notowania nie zapisywaliśmy nigdzie, więc nie dało się tego odsiać.
-- `stale_quote = true` znaczy: predykcja powstała na notowaniu sprzed ostatniej
-- rozpoczętej sesji albo rozliczenie użyło notowania sprzed pierwszej sesji po
-- predykcji. Taki wiersz wypada z trafności (raport, Track Record, recap,
-- kalibracja, `evaluate`).
--
-- NULL = nieznane: wiersze sprzed tej migracji, krypto (handel 24/7, brak sesji)
-- i zapis bez flagi. Bez backfillu — czasu notowania starych wierszy nie ma
-- w bazie, więc nie da się go odtworzyć.
--
-- Zapisuje Fast Loop za flagą `stale_quote_enabled` (przed migracją nowy klucz
-- → PGRST204 i śmierć zapisu całej predykcji); odczyty trafności wybierają
-- kolumnę też tylko za flagą. Kolumna w istniejącej tabeli — RLS z 038 ją
-- obejmuje, uprawnienia się nie zmieniają.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE prediction_logs
    -- DAT-14: predykcja albo rozliczenie na nieaktualnym notowaniu; NULL = nieznane.
    ADD COLUMN IF NOT EXISTS stale_quote BOOLEAN;

COMMENT ON COLUMN prediction_logs.stale_quote IS
    'Predykcja na notowaniu sprzed ostatniej sesji NYSE albo rozliczenie na notowaniu sprzed sesji po predykcji (DAT-14). true = poza trafnością; NULL = nieznane (krypto, wiersze sprzed 055).';
