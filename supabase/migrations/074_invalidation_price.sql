-- =====================================================================
-- 074_invalidation_price.sql — poziom unieważnienia tezy predykcji (ML-27).
-- Uruchom po migracjach o niższych numerach (ręczny workflow „DB Migrate").
--
-- PO CO: mail mówi, co model myśli, ale nie kiedy przestać mu wierzyć. Za flagą
-- `invalidation_level_enabled` prompt predykcji prosi o poziom ceny, którego
-- przekroczenie obala tezę („BULLISH upada poniżej 118”). Domena waliduje go przy
-- normalizacji odpowiedzi (BULLISH pod ceną, BEARISH nad nią, najdalej 15%), a watch
-- (`main_watch.py`) alarmuje „Teza X unieważniona”, gdy cena go przekroczy:
--
--   * `invalidation_price` — poziom po walidacji w skali pozostałych cen
--     (DECIMAL(14,4)); NULL = flaga off, poziom odrzucony albo wiersz sprzed 074;
--   * `invalidation_alerted_at` — chwila alarmu watcha; jeden alarm na predykcję
--     (watch czyta tylko wiersze z NULL i stempluje je po wysyłce).
--
-- Zapisuje Fast Loop (save) i Shock Watch (stempel alarmu) za flagą; przed migracją
-- nowy klucz → PGRST204, a preflight FIN-05 zatrzymuje cykl przed płatnymi węzłami.
-- Kolumny w istniejącej tabeli — RLS z 038 je obejmuje, uprawnienia się nie zmieniają.
-- Indeks częściowy obsługuje odczyt watcha: otwarte predykcje z poziomem bez alarmu.
--
-- Idempotentne: ADD COLUMN IF NOT EXISTS, CHECK w ADD COLUMN (nie dubluje się przy
-- powtórce), CREATE INDEX IF NOT EXISTS, COMMENT można powtarzać.
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS invalidation_price DECIMAL(14, 4) CHECK (invalidation_price > 0),
    ADD COLUMN IF NOT EXISTS invalidation_alerted_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_prediction_logs_open_invalidation
    ON prediction_logs (timestamp DESC)
    WHERE invalidation_price IS NOT NULL
      AND actual_price_after_12h IS NULL
      AND invalidation_alerted_at IS NULL;

COMMENT ON COLUMN prediction_logs.invalidation_price IS
    'ML-27: poziom ceny obalający tezę predykcji (BULLISH pod ceną, BEARISH nad nią, '
    'najdalej 15%). NULL = brak poprawnego poziomu albo flaga off.';

COMMENT ON COLUMN prediction_logs.invalidation_alerted_at IS
    'ML-27: chwila alarmu watcha „Teza unieważniona”; jeden alarm na predykcję.';
