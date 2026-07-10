-- =====================================================================
-- 024_attestation.sql — commit-reveal weryfikowalny track record (#16)
-- Uruchom w Supabase SQL Editor PO 023_positions.sql.
--
-- PO CO: każdy publiczny track record można oskarżyć o backfill i survivorship
-- bias — sceptyk nie ma dziś ŻADNEGO dowodu, że predykcja powstała PRZED ruchem
-- ceny. Schemat commit-reveal to naprawia:
--
--   * COMMIT (w chwili predykcji): publikujemy SHA-256 z kanonicznego JSON-a
--     predykcji + losowej SOLI. Zapisujemy `commitment_hash`. Sam hash jest
--     jednokierunkowy i BEZ SOLI NIEODWRACALNY — nie zdradza treści predykcji,
--     a mimo to wiąże nas z nią (nie da się jej później podmienić).
--   * REVEAL (przy zamknięciu predykcji): DOPIERO WTEDY ujawniamy plaintext i
--     `commitment_salt`. Każdy przelicza SHA-256(plaintext+salt) i porównuje z
--     wcześniej opublikowanym hashem → dowód, że predykcja jest sprzed ruchu.
--
-- Dlatego SÓL jest tu osobną kolumną ujawnianą DOPIERO na etapie reveal — przed
-- nim hash bez soli jest praktycznie nieodwracalny (brute-force po soli odpada).
--
-- Uczciwość: daty commitów git są fałszowalne — claim to "tamper-evidence przy
-- branch protection", nie "niepodrabialny timestamp".
--
-- Obie kolumny NULLABLE: stare wiersze i predykcje bez attestacji (flaga
-- attestation_enabled=false) renderują się jak dziś; INSERT bez nich działa.
-- Idempotentne: ADD COLUMN IF NOT EXISTS.
-- =====================================================================

ALTER TABLE prediction_logs
    -- SHA-256 commitmentu; publikowany w chwili predykcji (bez soli = nieodwracalny).
    ADD COLUMN IF NOT EXISTS commitment_hash TEXT,
    -- Losowa sól; ujawniana DOPIERO przy reveal (zamknięciu predykcji).
    ADD COLUMN IF NOT EXISTS commitment_salt TEXT;

-- Sceptyk weryfikuje po HASHU (szuka commitmentu, potem sprawdza reveal) —
-- indeks po commitment_hash obsługuje ten wzorzec wyszukiwania wprost.
CREATE INDEX IF NOT EXISTS idx_prediction_logs_commitment_hash
    ON prediction_logs (commitment_hash);

COMMENT ON COLUMN prediction_logs.commitment_hash IS
    'SHA-256 commitmentu predykcji (#16); publikowany przy predykcji, bez soli nieodwracalny.';
COMMENT ON COLUMN prediction_logs.commitment_salt IS
    'Losowa sól commitmentu; ujawniana DOPIERO przy reveal (zamknięciu predykcji).';
