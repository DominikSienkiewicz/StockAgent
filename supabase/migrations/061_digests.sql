-- =====================================================================
-- 061_digests.sql — prywatne archiwum digestów (MAIL-24, zastępuje U3 / BEZ-27).
-- Uruchom PO 059_settlement_session_close.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: Gmail przycina mail powyżej ~102 KB, a web digest U3 publikowałby pełny
-- prywatny raport publicznie. Pełny HTML cyklu trafia teraz do PRYWATNEGO bucketu
-- Supabase Storage `digests`, a mail niesie podpisany URL z TTL 7 dni („Otwórz
-- w przeglądarce”). Ta tabela to indeks archiwum: jeden wiersz na cykl z tematem,
-- leadem i czołówką sygnałów — baza pod porównanie „co mówiliśmy tydzień temu”.
--
-- Kolumny:
--   cycle_id         — identyfikator cyklu i nazwa obiektu w buckecie
--                      (`<start UTC>-<run_id>`, np. 20261015T0530Z-18234567890);
--                      bez PII i bez tickerów, unikalny (upsert przy ponowieniu runu);
--   subject          — temat maila właściciela;
--   lead_json        — pozycje leadu (ikona, nagłówek, ticker);
--   top_signals_json — czołówka rankingu sygnałów (ticker, kierunek, pewność);
--   sent_at          — chwila złożenia archiwum (tuż przed wysyłką maila).
-- Świadomie BEZ kwot portfela: te są tylko w samym HTML w prywatnym buckecie.
--
-- Zapis za flagą `digest_archive_enabled` (config.toml, domyślnie false).
--
-- Uprawnienia (od 038): RLS bez polityk — klucz publiczny nie widzi wierszy,
-- service_role (BYPASSRLS) zapisuje i czyta. Bucket `digests` jest prywatny
-- (`public = false`) i bez polityk na `storage.objects`: plik czyta tylko klucz
-- serwisowy albo posiadacz podpisanego URL-a.
--
-- Bucket: schemat `storage` istnieje tylko w projekcie Supabase. Lokalny Postgres
-- testów go nie ma, więc wiersz bucketu wstawiamy warunkowo (EXECUTE w bloku DO —
-- bez schematu nic się nie parsuje). Gdy wstawienie się nie uda (brak uprawnień
-- roli migracji do `storage.buckets`), bucket zakłada się ręcznie: Dashboard →
-- Storage → New bucket → `digests`, Public bucket: wyłączone.
--
-- Idempotentne (IF NOT EXISTS, ON CONFLICT, ENABLE RLS można powtarzać).
-- =====================================================================

CREATE TABLE IF NOT EXISTS digests (
    id               BIGSERIAL PRIMARY KEY,
    cycle_id         TEXT NOT NULL CONSTRAINT digests_cycle_id_unique UNIQUE,
    subject          TEXT NOT NULL,
    lead_json        JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_signals_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    sent_at          TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Archiwum czyta się od najnowszego cyklu („co mówiliśmy tydzień temu”).
CREATE INDEX IF NOT EXISTS idx_digests_sent_at
    ON digests (sent_at DESC);

ALTER TABLE digests ENABLE ROW LEVEL SECURITY;

COMMENT ON TABLE digests IS
    'Indeks prywatnego archiwum digestów (MAIL-24): temat, lead i czołówka sygnałów per cykl; HTML w prywatnym buckecie Storage digests.';
COMMENT ON COLUMN digests.cycle_id IS
    'Identyfikator cyklu i nazwa obiektu <cycle_id>.html w buckecie digests — start UTC i numer runu, bez PII.';

DO $$
BEGIN
    IF to_regclass('storage.buckets') IS NOT NULL THEN
        EXECUTE $bucket$
            INSERT INTO storage.buckets (id, name, public)
            VALUES ('digests', 'digests', false)
            ON CONFLICT (id) DO UPDATE SET public = false
        $bucket$;
    END IF;
END
$$;
