"""Proweniencja treningu modelu (ML-22) — na jakich danych i z jakimi parametrami
powstał model, który wydał prognozę.

Pierwsze pytanie recenzenta ML i warunek uczciwego championa/challengera: karta
modelu i sidecar `.json` obok pliku wag niosą odcisk zbioru (sha256 po wierszach
niezależny od ich kolejności), zakres dat, liczności per symbol, parametry
z jawnym ziarnem, wersję biblioteki, wersję zestawu cech i SHA kodu. Ten sam
odcisk i te same parametry mają dać identyczny booster (narzędzie
`main_tools.py reproduce-training`).

Czysta domena: stdlib, bez I/O. Odcisk liczy adapter z wierszy, na których
faktycznie trenował; resztę opisu próbki zna wołający (`DatasetDescription`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.domain.backtest import LabelOverlap

# Kolejność kluczy rekordu = kolejność kolumn karty (`model_scorecards`, migracja 064)
# i pól sidecara. Jedno źródło dla adaptera bazy i pliku.
PROVENANCE_RECORD_KEYS: tuple[str, ...] = (
    "dataset_sha256",
    "data_start",
    "data_end",
    "n_per_symbol",
    "training_params",
    "library_version",
    "feature_set_version",
    "git_sha",
    "label_overlap_count",
    "label_overlap_measured",
)

TrainingRow = tuple[datetime | None, float, Mapping[str, float | None]]


def _canonical_number(value: float | None) -> str | None:
    """Liczba jako dokładny `repr` floata; brak i NaN to ta sama luka (`None`)."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else repr(number)


def _canonical_moment(moment: datetime | None) -> str | None:
    """Chwila w UTC (naiwna czytana jako UTC) — ta sama chwila w innej strefie
    to ten sam wiersz."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def _canonical_row(row: TrainingRow) -> str:
    moment, target, features = row
    return json.dumps(
        [
            _canonical_moment(moment),
            _canonical_number(target),
            sorted((str(name), _canonical_number(value)) for name, value in features.items()),
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def dataset_fingerprint(rows: Iterable[TrainingRow]) -> str:
    """sha256 (hex) zbioru treningu z wierszy `(chwila, target, cechy)`.

    Wiersz to chwila (UTC), target i cechy posortowane po nazwie; linie wierszy są
    sortowane przed haszowaniem, więc kolejność wejścia (wierszy i kluczy cech)
    nie zmienia odcisku, a zmiana dowolnej wartości albo duplikat wiersza — tak.
    Liczby idą jako dokładny `repr` floata (1 i 1.0 to ta sama wartość); brak
    cechy i NaN to ta sama luka.
    """
    lines = sorted(_canonical_row(row) for row in rows)
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DatasetDescription:
    """Co o próbce treningu wie wołający, a czego nie widzi adapter modelu.

    `n_per_symbol` — liczba wierszy treningu per symbol (None = nieznana),
    `feature_set_version` — wersja zestawu cech próbki (None = cała historia),
    `git_sha` — SHA kodu przebiegu (`RunContext.git_sha`, lokalnie „local”),
    `label_overlap` — ML-25: ile etykiet dojrzało po następnym wierszu symbolu
    (None = próbka bez chwili rozliczenia).
    """

    n_per_symbol: Mapping[str, int] | None = None
    feature_set_version: int | None = None
    git_sha: str | None = None
    label_overlap: LabelOverlap | None = None


@dataclass(frozen=True)
class TrainingProvenance:
    """Pełna proweniencja jednego treningu: odcisk i zakres danych policzone przez
    adapter z wierszy, na których trenował, parametry modelu z jawnym ziarnem,
    wersja biblioteki i opis próbki od wołającego."""

    dataset_sha256: str
    data_start: datetime | None
    data_end: datetime | None
    params: Mapping[str, object]
    library_version: str
    dataset: DatasetDescription = field(default_factory=DatasetDescription)

    def as_record(self) -> dict[str, object]:
        """Rekord karty i sidecara (klucze `PROVENANCE_RECORD_KEYS`, wartości
        serializowalne do JSON); symbole posortowane, żeby plik był stabilny."""
        n_per_symbol = self.dataset.n_per_symbol
        overlap = self.dataset.label_overlap
        return {
            "dataset_sha256": self.dataset_sha256,
            "data_start": _canonical_moment(self.data_start),
            "data_end": _canonical_moment(self.data_end),
            "n_per_symbol": (
                None if n_per_symbol is None else dict(sorted(n_per_symbol.items()))
            ),
            "training_params": dict(self.params),
            "library_version": self.library_version,
            "feature_set_version": self.dataset.feature_set_version,
            "git_sha": self.dataset.git_sha,
            "label_overlap_count": None if overlap is None else overlap.overlapping,
            "label_overlap_measured": None if overlap is None else overlap.measured,
        }
