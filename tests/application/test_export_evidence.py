"""NIEZ-32: tygodniowy eksport tabel dowodowych (track record) jako kopia zapasowa.

Plan Free Supabase nie ma PITR — utrata projektu albo migracja z DROP kasuje cały
dowód. Eksport bierze WYŁĄCZNIE kolumny z listy dozwolonych: bez embeddingów, bez
soli commitmentów (nie może wyciec przed revealem), bez pozycji i subskrybentów.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

from src.application.ports import ArchiveReceipt, EvidenceArchivePort, EvidenceSourcePort
from src.application.use_cases.export_evidence import (
    EVIDENCE_COLUMNS,
    MANIFEST_NAME,
    ExportEvidenceUseCase,
    archive_name,
)

NOW = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
SENSITIVE = ("embedding", "commitment_salt", "news_summary", "email")


class _FakeSource(EvidenceSourcePort):
    """Baza, która oddaje więcej, niż ją poproszono — eksport ma to odsiać."""

    def __init__(self) -> None:
        self.requested: dict[str, tuple[str, ...]] = {}

    def read_rows(self, table: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        self.requested[table] = tuple(columns)
        leaky = {
            "embedding": [0.1] * 4,
            "commitment_salt": "s3cr3t-salt-value",
            "news_summary": "news",
            "email": "jan.kowalski@example.com",
        }
        if table == "prediction_logs":
            return [
                {
                    "id": "p1",
                    "symbol": "NVDA",
                    "predicted_trend": "BULLISH",
                    "price_at_prediction": Decimal("120.50"),
                    "reasoning_text": "Pisz do mnie: jan.kowalski@example.com",
                    "council_verdict": {"final_recommendation": "BUY"},
                    "commitment_hash": "abc123",
                    **leaky,
                }
            ]
        if table == "council_votes":
            return [{"id": 1, "investor_name": "Buffett", "key_factors": ["moat"], **leaky}]
        return [{"id": 7, "status": "deployed", "n_samples": 300, **leaky}]


def _files() -> dict[str, bytes]:
    return ExportEvidenceUseCase(_FakeSource(), Mock(spec=EvidenceArchivePort)).build(now=NOW)


def test_export_excludes_sensitive_columns() -> None:
    files = _files()

    exported = b"".join(files.values()).decode("utf-8")
    for column in SENSITIVE:
        assert column not in exported
    assert "s3cr3t-salt-value" not in exported
    assert "@" not in exported


def test_only_allowlisted_columns_are_requested() -> None:
    source = _FakeSource()

    ExportEvidenceUseCase(source, Mock(spec=EvidenceArchivePort)).build(now=NOW)

    assert set(source.requested) == {"prediction_logs", "council_votes", "model_scorecards"}
    for table, columns in source.requested.items():
        assert columns == EVIDENCE_COLUMNS[table]
        assert not set(columns) & set(SENSITIVE)


def test_each_table_is_one_csv_with_its_allowlisted_header() -> None:
    files = _files()

    rows = list(csv.DictReader(io.StringIO(files["prediction_logs.csv"].decode("utf-8"))))
    assert list(rows[0]) == list(EVIDENCE_COLUMNS["prediction_logs"])
    assert rows[0]["price_at_prediction"] == "120.50"
    assert json.loads(rows[0]["council_verdict"]) == {"final_recommendation": "BUY"}
    assert rows[0]["confidence_score"] == ""
    assert "[e-mail usunięty]" in rows[0]["reasoning_text"]


def test_manifest_counts_rows_per_table() -> None:
    manifest = json.loads(_files()[MANIFEST_NAME])

    assert manifest["created_at"] == NOW.isoformat()
    assert manifest["tables"]["prediction_logs"]["rows"] == 1
    assert manifest["tables"]["council_votes"]["columns"] == list(EVIDENCE_COLUMNS["council_votes"])


def test_run_hands_the_files_to_the_encrypting_archive() -> None:
    archive = Mock(spec=EvidenceArchivePort)
    archive.store.return_value = ArchiveReceipt(
        path="data/backup/evidence-2026-W43.tar.gz.age", sha256="ab" * 32, size_bytes=1234
    )

    receipt = ExportEvidenceUseCase(_FakeSource(), archive).run(now=NOW)

    assert receipt.sha256 == "ab" * 32
    name, files = archive.store.call_args.args
    assert name == "evidence-2026-W43"
    assert set(files) == {
        "prediction_logs.csv",
        "council_votes.csv",
        "model_scorecards.csv",
        MANIFEST_NAME,
    }


def test_archive_name_is_the_iso_week() -> None:
    assert archive_name(datetime(2027, 1, 1, tzinfo=UTC)) == "evidence-2026-W53"
    assert archive_name(datetime(2026, 1, 5, tzinfo=UTC)) == "evidence-2026-W02"
