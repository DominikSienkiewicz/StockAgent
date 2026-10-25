-- =====================================================================
-- 062_corporate_actions.sql — podejrzenia splitów wykryte z `pc` notowania
-- (DAT-23). Uruchom workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: split 10:1 bez korekty to dla systemu ruch −90% — reflect oceniłby go
-- jako trafny BEARISH, krzywa kapitału pokazałaby zysk, a cecha `price_delta`
-- zatrułaby trening. Cykl Fast Loopa idzie przed otwarciem, więc `pc` z Finnhub
-- /quote to zamknięcie sprzed ostatniej sesji, czyli cena poprzedniego snapshotu.
-- Stosunek `pc / snapshot` bliski stosunkowi splitu (±2%) to korekta historii
-- przez dostawcę, nie ruch ceny: cykl rozlicza wtedy predykcję w jednostkach
-- sprzed splitu i zapisuje tu zdarzenie (audyt, późniejsza korekta cech).
--
-- Kolumny:
--   kind            — rodzaj zdarzenia (dziś tylko 'split_suspected');
--   ratio           — mnożnik ceny po splicie względem ceny przed nim
--                     (10:1 → 0.1, odwrotny 1:2 → 2);
--   prev_close      — `pc` notowania, które ujawniło split;
--   reference_price — ostatni snapshot ceny (sprzed splitu);
--   detected_at     — chwila cyklu; detected_on — jej dzień UTC, klucz idempotencji
--                     (powtórka cyklu tego dnia nie dubluje wiersza).
--
-- Zapis za flagą `split_detection_enabled` (przed migracją tabeli nie ma).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS i COMMENT można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS corporate_actions (
    id               BIGSERIAL PRIMARY KEY,
    symbol           TEXT NOT NULL,
    kind             TEXT NOT NULL CONSTRAINT corporate_actions_kind_known
                         CHECK (kind IN ('split_suspected')),
    ratio            NUMERIC NOT NULL CONSTRAINT corporate_actions_ratio_positive
                         CHECK (ratio > 0),
    prev_close       DECIMAL(14, 4) NOT NULL,
    reference_price  DECIMAL(14, 4) NOT NULL,
    detected_at      TIMESTAMPTZ NOT NULL,
    detected_on      DATE NOT NULL,
    CONSTRAINT corporate_actions_unique_event UNIQUE (symbol, kind, detected_on)
);

ALTER TABLE corporate_actions ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE corporate_actions IS
    'Podejrzenia splitów z pc notowania vs ostatni snapshot (DAT-23): jeden wiersz na symbol, rodzaj i dzień UTC.';

COMMENT ON COLUMN corporate_actions.ratio IS
    'Mnożnik ceny po splicie względem ceny przed nim (10:1 → 0.1, odwrotny 1:2 → 2).';
