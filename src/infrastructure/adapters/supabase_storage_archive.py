"""MAIL-24: prywatne archiwum digestów w Supabase Storage (zastępuje U3, BEZ-27).

HTML cyklu trafia do PRYWATNEGO bucketu `digests` (bez polityk — czyta tylko
klucz serwisowy), a mail dostaje podpisany URL z TTL 7 dni: pełna wersja jest
dostępna mimo przycinania Gmaila, bez publicznej ścieżki Pages. Signed URL działa
jak token na okaziciela — dlatego krótki TTL i ścieżka z samego identyfikatora
cyklu (data i numer runu, zero PII i tickerów).

Indeks archiwum to tabela `digests` (migracja 061): temat, lead i czołówka
sygnałów per cykl — baza pod porównanie „co mówiliśmy tydzień temu”. Każdy błąd
magazynu albo tabeli leci jako `PortError` (ARCH-33); wołający wysyła wtedy mail
bez linku.
"""

from __future__ import annotations

from typing import Any

import httpx
from postgrest.exceptions import APIError
from supabase import StorageException

from src.application.errors import MalformedPayload, ProviderUnavailable
from src.application.ports import DigestArchivePort, DigestRecord
from src.infrastructure.adapters._supabase import build_supabase_client
from src.infrastructure.error_translation import as_port_error

DIGESTS_BUCKET = "digests"
DIGESTS_TABLE = "digests"
# Decyzja 2026-10-24: link żyje tydzień — dłużej to token na okaziciela do portfela.
SIGNED_URL_TTL_SECONDS = 7 * 24 * 3600
_UPLOAD_OPTIONS = {"content-type": "text/html; charset=utf-8", "upsert": "true"}


def digest_object_path(cycle_id: str) -> str:
    """Ścieżka obiektu w buckecie: sam identyfikator cyklu, bez PII."""
    return f"{cycle_id}.html"


class SupabaseStorageDigestArchive(DigestArchivePort):
    """`client` to klient Supabase z `build_supabase_client` (klucz serwisowy, NIEZ-35)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str, key: str) -> SupabaseStorageDigestArchive:
        """Archiwum na nowym kliencie Supabase (klucz serwisowy, jak repozytorium)."""
        return cls(build_supabase_client(url, key))

    def store(self, cycle_id: str, html: str) -> str:
        path = digest_object_path(cycle_id)
        try:
            bucket = self._client.storage.from_(DIGESTS_BUCKET)
            bucket.upload(path, html.encode("utf-8"), dict(_UPLOAD_OPTIONS))
            signed = bucket.create_signed_url(path, expires_in=SIGNED_URL_TTL_SECONDS)
        except (StorageException, httpx.HTTPError) as exc:
            raise as_port_error(exc, ProviderUnavailable) from None
        url = signed.get("signedURL") or signed.get("signedUrl")
        if not url:
            raise MalformedPayload(f"Supabase Storage returned no signed URL for {path}")
        return str(url)

    def record(self, entry: DigestRecord) -> None:
        payload = {
            "cycle_id": entry.cycle_id,
            "subject": entry.subject,
            "lead_json": [dict(item) for item in entry.lead],
            "top_signals_json": [dict(item) for item in entry.top_signals],
            "sent_at": entry.sent_at.isoformat(),
        }
        try:
            self._client.table(DIGESTS_TABLE).upsert(payload, on_conflict="cycle_id").execute()
        except (APIError, httpx.HTTPError) as exc:
            raise as_port_error(exc, ProviderUnavailable) from None
