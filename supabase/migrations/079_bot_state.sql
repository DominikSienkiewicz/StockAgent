-- =====================================================================
-- 079_bot_state.sql — offset pollingu bota Telegrama (MAIL-48).
-- Uruchom PO 076, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: bot właściciela odpytuje Telegram Bot API (`getUpdates`) z crona GitHub
-- Actions co 15 minut, bez własnego serwera. Telegram oddaje każdą aktualizację,
-- dopóki kolejne wywołanie nie poda `offset` = ostatni `update_id` + 1. Runner GHA
-- nie ma stanu między przebiegami, więc offset żyje tutaj: przebieg czyta go na
-- starcie i zapisuje po obsłużeniu komend. Bez tabeli każdy przebieg odpowiadałby
-- na te same komendy od nowa (do 24 h, tyle Telegram trzyma aktualizacje).
--
-- Kolumny:
--   channel       — kanał bota; dziś wyłącznie 'telegram' (jeden wiersz);
--   update_offset — następny `update_id` do pobrania (ostatni obsłużony + 1);
--   updated_at    — chwila ostatniego zapisu offsetu.
-- Świadomie BEZ treści wiadomości i chat_id: komendy nie są archiwizowane.
--
-- Zapis i odczyt: `main_bot.py` za flagą `telegram_bot_enabled` (config.toml,
-- domyślnie false).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wiersza,
-- service_role (BYPASSRLS) czyta i zapisuje.
--
-- Idempotentne (IF NOT EXISTS, ENABLE RLS i COMMENT można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS bot_state (
    channel       TEXT PRIMARY KEY
                      CONSTRAINT bot_state_channel_known CHECK (channel IN ('telegram')),
    update_offset BIGINT NOT NULL
                      CONSTRAINT bot_state_offset_non_negative CHECK (update_offset >= 0),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE bot_state ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE bot_state IS
    'Offset pollingu getUpdates bota właściciela (MAIL-48): jeden wiersz na kanał, bez treści komend i chat_id.';
COMMENT ON COLUMN bot_state.update_offset IS
    'Następny update_id do pobrania z Telegram Bot API (ostatni obsłużony + 1).';
