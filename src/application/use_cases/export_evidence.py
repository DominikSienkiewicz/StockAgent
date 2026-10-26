"""NIEZ-32: tygodniowa kopia zapasowa tabel dowodowych (krok Slow Loopa).

Track record to główny dowód decyzji 1 („żywy, weryfikowalny dowód”), a plan
Free Supabase nie ma PITR: utrata projektu albo migracja z DROP kasuje go cały.
Eksport czyta kolumny z listy dozwolonych z `prediction_logs`, `council_votes`
i `model_scorecards`, składa CSV per tabela plus manifest i oddaje je magazynowi,
który szyfruje je przed zapisem (`EvidenceArchivePort`).

Lista dozwolonych, nie lista zakazanych: nowa kolumna nie wejdzie sama. Poza nią
zostają embeddingi, sól commitmentów (nie może wyciec przed revealem), tekst
newsów, a tabele pozycji i subskrybentów nie są czytane wcale. Adresy e-mail
w tekstach (np. w uzasadnieniu) są zastępowane znacznikiem.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.application.ports import ArchiveReceipt, EvidenceArchivePort, EvidenceSourcePort

MANIFEST_NAME = "manifest.json"
EMAIL_PLACEHOLDER = "[e-mail usunięty]"
_EMAIL = re.compile(r"[^\s@<>()\"',;:]+@[^\s@<>()\"',;:]+")

# Kolumny w kolejności migracji (komentarz = numer migracji, gdy późniejsza niż 001).
EVIDENCE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "prediction_logs": (
        "id",
        "symbol",
        "timestamp",
        "price_at_prediction",
        "sentiment_score",
        "predicted_trend",
        "predicted_target_price",
        "reasoning_text",
        "actual_price_after_12h",
        "accuracy_score",
        "correction_insights",
        "council_verdict",  # 005
        "data_quality_flags",  # 008
        "is_trend_correct",  # 009
        "confidence_score",  # 016
        "confidence_calibration",  # 016
        "calibration_insight",  # 016
        "regime",  # 017
        "decision_receipts",  # 020
        "alpha_fusion_score",  # 022
        "commitment_hash",  # 024 — sam hash; sól zostaje w bazie
        "revealed_at",  # 024
        "edge_sigma",  # 025
        "feature_set_version",  # 027
        "settled_at",  # 029
        "recipe_version",  # 030
        "interval_low",  # 032
        "interval_high",  # 032
        "gate_delta",  # 034
        "experiment_arms",  # 037
        "stale_quote",  # 055 — tylko ze `stale_quote_enabled` (NIEZ-50)
    ),
    "council_votes": (  # 007
        "id",
        "prediction_id",
        "symbol",
        "timestamp",
        "investor_name",
        "recommendation",
        "confidence",
        "reasoning",
        "key_factors",
    ),
    "model_scorecards": (  # 019
        "id",
        "symbol",
        "status",
        "candidate_holdout_rmse",
        "baseline_rmse",
        "candidate_holdout_directional_hit_rate",
        "n_folds",
        "n_samples",
        "timestamp",
        "drift_baseline_rmse",  # 028
        "llm_baseline_rmse",  # 028
    ),
}


# NIEZ-50: znacznik DAT-14 istnieje od migracji 055 i bez flagi nikt go nie zapisuje —
# eksport czyta go za `stale_quote_enabled`, jak odczyty trafności (ML-57).
STALE_QUOTE_COLUMN = "stale_quote"


def archive_name(now: datetime) -> str:
    """„evidence-RRRR-Wtt” — tydzień ISO przebiegu (jeden artefakt na tydzień)."""
    year, week, _ = now.isocalendar()
    return f"evidence-{year}-W{week:02d}"


def _without_emails(text: str) -> str:
    return _EMAIL.sub(EMAIL_PLACEHOLDER, text)


def _cell(value: Any) -> str:
    """Wartość komórki CSV: JSONB jako JSON, liczby i daty jako tekst, None pusto."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return _without_emails(json.dumps(value, ensure_ascii=False, sort_keys=True))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return _without_emails(str(value))


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(column)) for column in columns])
    return buffer.getvalue().encode("utf-8")


class ExportEvidenceUseCase:
    """Składa kopię tabel dowodowych i oddaje ją szyfrującemu magazynowi."""

    def __init__(
        self,
        source: EvidenceSourcePort,
        archive: EvidenceArchivePort,
        *,
        stale_quote_enabled: bool = True,
    ) -> None:
        """`stale_quote_enabled` (NIEZ-50) — False zostawia `stale_quote` poza odczytem
        (baza bez migracji 055). Domyślnie cała lista dozwolonych; Slow Loop podaje flagę
        z configu."""
        self._source = source
        self._archive = archive
        self._stale_quote_enabled = stale_quote_enabled

    def _columns(self, columns: tuple[str, ...]) -> tuple[str, ...]:
        if self._stale_quote_enabled:
            return columns
        return tuple(column for column in columns if column != STALE_QUOTE_COLUMN)

    def build(self, now: datetime) -> dict[str, bytes]:
        """CSV per tabela (tylko kolumny z `EVIDENCE_COLUMNS`) plus manifest."""
        files: dict[str, bytes] = {}
        tables: dict[str, dict[str, Any]] = {}
        for table, allowed in EVIDENCE_COLUMNS.items():
            columns = self._columns(allowed)
            rows = self._source.read_rows(table, columns)
            files[f"{table}.csv"] = _csv(columns, rows)
            tables[table] = {"rows": len(rows), "columns": list(columns)}
        manifest = {"created_at": now.isoformat(), "tables": tables}
        files[MANIFEST_NAME] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        return files

    def run(self, now: datetime) -> ArchiveReceipt:
        return self._archive.store(archive_name(now), self.build(now))
