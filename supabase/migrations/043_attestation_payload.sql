-- =====================================================================
-- 043_attestation_payload.sql — dosłowny plaintext commitmentu (WIA-70).
-- Uruchom PO 042_reference_predictions.sql, workflow „🗄️ DB Migrate (manual)”.
--
-- PO CO: reveal z 024 odtwarzał payload z kolumn, a kolumny nie oddają bajtów,
-- z których powstał hash — PostgREST zwraca NUMERIC jako float (100.0 zamiast
-- "100.0000"), a pewność i werdykt rady żyją w innych typach. `commitment_payload`
-- trzyma kanoniczny JSON (`attestation.canonicalize`) dokładnie w tej postaci,
-- którą zahashowano, więc sceptyk liczy SHA-256(payload + sól) bez żadnej
-- rekonstrukcji. TEXT, nie JSONB: JSONB przestawia klucze i białe znaki.
--
-- NULLABLE: stare wiersze i predykcje bez atestacji (flaga attestation_enabled)
-- zostają z NULL; sweep ujawnia je z kolumn, jak przed 043.
--
-- Zapis za flagą `attestation_enabled`; Fast Loop sprawdza kolumnę przed cyklem
-- (`attestation_columns_available`) i bez niej jedzie bez atestacji.
--
-- Uprawnienia (od 038): nowa kolumna w istniejącej tabeli — RLS `prediction_logs`
-- obowiązuje bez zmian.
--
-- Idempotentne (ADD COLUMN IF NOT EXISTS, COMMENT można powtarzać).
-- =====================================================================

ALTER TABLE prediction_logs
    ADD COLUMN IF NOT EXISTS commitment_payload TEXT;

COMMENT ON COLUMN prediction_logs.commitment_payload IS
    'Kanoniczny JSON commitmentu (#16, WIA-70) dokładnie w postaci zahashowanej; '
    'ujawniany dosłownie przy reveal. NULL = wiersz sprzed 043 albo bez atestacji.';
