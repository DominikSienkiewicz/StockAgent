"""MAIL-24: prywatne archiwum digestów w Supabase Storage (bucket `digests`)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest
from postgrest.exceptions import APIError
from supabase import StorageException

from src.application.errors import MalformedPayload, PortError
from src.application.ports import DigestRecord
from src.infrastructure.adapters.supabase_storage_archive import (
    DIGESTS_BUCKET,
    SIGNED_URL_TTL_SECONDS,
    SupabaseStorageDigestArchive,
    digest_object_path,
)

SIGNED = "https://proj.supabase.co/storage/v1/object/sign/digests/20261015T0530Z-1823.html?token=t"


def _client(signed: dict[str, str | None] | None = None) -> MagicMock:
    client = MagicMock()
    bucket = client.storage.from_.return_value
    bucket.create_signed_url.return_value = (
        {"signedURL": SIGNED, "signedUrl": SIGNED} if signed is None else signed
    )
    return client


def test_store_uploads_private_and_returns_signed_url() -> None:
    client = _client()

    url = SupabaseStorageDigestArchive(client).store("20261015T0530Z-1823", "<html>ż</html>")

    client.storage.from_.assert_called_with(DIGESTS_BUCKET)
    bucket = client.storage.from_.return_value
    path, body, options = bucket.upload.call_args.args
    assert path == "20261015T0530Z-1823.html"
    assert body == "<html>ż</html>".encode()
    assert options == {"content-type": "text/html; charset=utf-8", "upsert": "true"}
    bucket.create_signed_url.assert_called_once_with("20261015T0530Z-1823.html", expires_in=604800)
    assert SIGNED_URL_TTL_SECONDS == 604800
    assert url == SIGNED


def test_object_path_is_the_cycle_id_without_personal_data() -> None:
    assert digest_object_path("20261015T0530Z-local") == "20261015T0530Z-local.html"


@pytest.mark.parametrize(
    "error",
    [
        StorageException("Bucket not found"),
        httpx.ConnectError("connection reset"),
    ],
)
def test_storage_failure_is_a_port_error(error: Exception) -> None:
    client = _client()
    client.storage.from_.return_value.upload.side_effect = error

    with pytest.raises(PortError):
        SupabaseStorageDigestArchive(client).store("c1", "<html>")


def test_signed_url_without_a_link_is_malformed() -> None:
    client = _client({"signedURL": None, "signedUrl": None})

    with pytest.raises(MalformedPayload):
        SupabaseStorageDigestArchive(client).store("c1", "<html>")


def test_record_upserts_the_index_row_by_cycle() -> None:
    client = MagicMock()
    entry = DigestRecord(
        cycle_id="20261015T0530Z-1823",
        subject="StockAgent — NVDA -4,2%",
        lead=({"icon": "🔥", "headline": "NVDA -4,2%", "symbol": "NVDA"},),
        top_signals=({"symbol": "NVDA", "direction": "SELL", "confidence": 0.8},),
        sent_at=datetime(2026, 10, 15, 5, 31, tzinfo=UTC),
    )

    SupabaseStorageDigestArchive(client).record(entry)

    client.table.assert_called_once_with("digests")
    payload = client.table.return_value.upsert.call_args.args[0]
    assert payload == {
        "cycle_id": "20261015T0530Z-1823",
        "subject": "StockAgent — NVDA -4,2%",
        "lead_json": [{"icon": "🔥", "headline": "NVDA -4,2%", "symbol": "NVDA"}],
        "top_signals_json": [{"symbol": "NVDA", "direction": "SELL", "confidence": 0.8}],
        "sent_at": "2026-10-15T05:31:00+00:00",
    }
    assert client.table.return_value.upsert.call_args.kwargs == {"on_conflict": "cycle_id"}


def test_index_write_failure_is_a_port_error() -> None:
    client = MagicMock()
    client.table.return_value.upsert.return_value.execute.side_effect = APIError(
        {"message": "relation digests does not exist", "code": "42P01"}
    )
    entry = DigestRecord("c1", "s", (), (), datetime(2026, 10, 15, tzinfo=UTC))

    with pytest.raises(PortError):
        SupabaseStorageDigestArchive(client).record(entry)
