-- =====================================================================
-- 080_symbol_mutes.sql — wyciszenia symboli z bota właściciela (MAIL-45).
-- Uruchom PO 079_bot_state.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: zmiana watchlisty wymaga commita do config.toml, więc właśnie sprzedanej
-- spółki nie dało się wyciszyć bez deployu. `/mute NVDA 14` w bocie zapisuje tu
-- wyciszenie do chwili `muted_until`; Fast Loop pomija wyciszony symbol w płatnej
-- analizie i w sekcjach maila, ale zapisuje jego snapshot ceny (punkt odniesienia
-- po powrocie). Wygasłe wyciszenie wraca symbol samo — bez zapisu.
--
-- Decyzja 2026-10-23 (opcja b): baza trzyma TYLKO wyciszenia. Symbole dodaje się
-- wyłącznie przez config.toml — ta tabela nie jest drugim źródłem prawdy watchlisty,
-- a wyciszenie symbolu spoza config.toml niczego nie dodaje.
--
-- Kolumny:
--   symbol      — ticker wielkimi literami (jak w config.toml), jeden wiersz na symbol;
--   muted_until — koniec wyciszenia (wyłącznie); ponowne /mute nadpisuje;
--   created_at  — chwila ostatniego /mute tego symbolu.
-- `/unmute` usuwa wiersz.
--
-- Zapis: `main_bot.py`, odczyt: `main_bot.py` i Fast Loop — oba za flagą
-- `symbol_mutes_enabled` (config.toml, domyślnie false).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) czyta, zapisuje i usuwa.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS i COMMENT można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS symbol_mutes (
    symbol      TEXT PRIMARY KEY
                    CONSTRAINT symbol_mutes_symbol_upper
                    CHECK (symbol <> '' AND symbol = upper(symbol)),
    muted_until TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE symbol_mutes ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE symbol_mutes IS
    'Wyciszenia symboli z bota właściciela (MAIL-45): Fast Loop pomija symbol do muted_until; symbole dodaje wyłącznie config.toml.';
COMMENT ON COLUMN symbol_mutes.muted_until IS
    'Koniec wyciszenia (wyłącznie): od tej chwili symbol wraca do analizy bez żadnego zapisu.';
