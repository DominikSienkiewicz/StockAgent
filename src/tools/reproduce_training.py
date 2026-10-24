"""ML-22: czy odcisk zbioru i ziarno z sidecara wag dają zapisany booster (czyste funkcje).

Sidecar `.json` obok pliku wag niesie odcisk zbioru treningu, zakres dat
i parametry z jawnym ziarnem. Narzędzie odbudowuje próbkę modelu wspólnego tą samą
ścieżką co Slow Loop (`TrainModelUseCase.pooled_training_set`), zostawia wiersze
z zakresu dat sidecara, liczy odcisk tą samą funkcją domeny co adapter i — gdy
zbiór się zgadza — porównuje zrzut drzew refitu ze zrzutem zapisanych wag.
Odczyt bazy, refit i kod wyjścia należą do `main_tools.py reproduce-training`
(ARCH-39); ten moduł nie robi I/O.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from src.application.use_cases.train_model import TrainingSet
from src.domain.training_provenance import dataset_fingerprint


class SidecarError(ValueError):
    """Sidecar bez pól, z których da się odtworzyć trening."""


@dataclass(frozen=True)
class Sidecar:
    """Pola sidecara potrzebne do odtworzenia treningu."""

    dataset_sha256: str
    data_start: datetime | None
    data_end: datetime | None
    params: Mapping[str, object]
    library_version: str | None
    git_sha: str | None


def _moment(record: Mapping[str, object], key: str) -> datetime | None:
    raw = record.get(key)
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise SidecarError(f"{key} is not an ISO timestamp: {raw!r}") from exc


def _optional_text(record: Mapping[str, object], key: str) -> str | None:
    raw = record.get(key)
    return None if raw is None else str(raw)


def parse_sidecar(text: str) -> Sidecar:
    """Sidecar z tekstu JSON; brak odcisku albo parametrów → `SidecarError`."""
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SidecarError(f"sidecar is not JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise SidecarError("sidecar is not a JSON object")
    digest = record.get("dataset_sha256")
    params = record.get("training_params")
    if not isinstance(digest, str):
        raise SidecarError("sidecar has no dataset_sha256")
    if not isinstance(params, dict):
        raise SidecarError("sidecar has no training_params object")
    return Sidecar(
        dataset_sha256=digest,
        data_start=_moment(record, "data_start"),
        data_end=_moment(record, "data_end"),
        params=params,
        library_version=_optional_text(record, "library_version"),
        git_sha=_optional_text(record, "git_sha"),
    )


def rows_in_range(training_set: TrainingSet, sidecar: Sidecar) -> TrainingSet:
    """Wiersze próbki z zakresu dat sidecara (domknięty z obu stron; brak granicy
    = bez cięcia z tej strony). Późniejsze przebiegi widzą wiersze, których etykieta
    dojrzała po treningu — zakres je odcina."""
    keep = [
        index
        for index, moment in enumerate(training_set.timestamps)
        if (sidecar.data_start is None or moment >= sidecar.data_start)
        and (sidecar.data_end is None or moment <= sidecar.data_end)
    ]
    ends = training_set.label_ends
    return dataclasses.replace(
        training_set,
        features=[training_set.features[index] for index in keep],
        target=[training_set.target[index] for index in keep],
        timestamps=[training_set.timestamps[index] for index in keep],
        label_ends=None if ends is None else [ends[index] for index in keep],
    )


def fingerprint_of(training_set: TrainingSet) -> str:
    """Odcisk próbki — ta sama funkcja domeny, którą liczy go adapter przy treningu."""
    return dataset_fingerprint(
        zip(training_set.timestamps, training_set.target, training_set.features, strict=True)
    )


@dataclass(frozen=True)
class ReproductionReport:
    """Werdykt: zgodność zbioru, zgodność boostera (None = nie porównywano, bo
    zbiór się różni) i kontekst do wydruku."""

    sidecar: Sidecar
    fingerprint: str
    n_rows: int
    dump_matches: bool | None
    library_version: str

    @property
    def dataset_matches(self) -> bool:
        return self.fingerprint == self.sidecar.dataset_sha256

    @property
    def ok(self) -> bool:
        return self.dataset_matches and self.dump_matches is True


def assess_reproduction(
    sidecar: Sidecar,
    *,
    fingerprint: str,
    n_rows: int,
    refit_dump: str | None,
    saved_dump: str,
    library_version: str,
) -> ReproductionReport:
    """Werdykt odtworzenia; `refit_dump` None = refitu nie robiono (inny zbiór)."""
    return ReproductionReport(
        sidecar=sidecar,
        fingerprint=fingerprint,
        n_rows=n_rows,
        dump_matches=None if refit_dump is None else refit_dump == saved_dump,
        library_version=library_version,
    )


def format_reproduction(report: ReproductionReport) -> str:
    """Wydruk werdyktu: REPRODUCED albo krok, na którym odtworzenie się rozjechało."""
    sidecar = report.sidecar
    lines = [
        f"Sidecar: dataset {sidecar.dataset_sha256[:12]}, commit {sidecar.git_sha or 'n/a'}, "
        f"{sidecar.library_version or 'library n/a'}",
        f"Rebuilt sample: {report.n_rows} rows, dataset {report.fingerprint[:12]}",
    ]
    if sidecar.library_version is not None and sidecar.library_version != report.library_version:
        lines.append(
            f"Warning: trained with {sidecar.library_version}, refit with "
            f"{report.library_version} — the booster may differ for that reason alone."
        )
    if not report.dataset_matches:
        lines.append(
            "NOT REPRODUCED: dataset differs (rows added, removed or changed since the "
            "training) — the booster was not refit."
        )
    elif report.dump_matches:
        lines.append("REPRODUCED: same dataset and seed give an identical booster dump.")
    else:
        lines.append("NOT REPRODUCED: same dataset, but the refit booster differs.")
    return "\n".join(lines)
