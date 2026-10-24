"""WIA-45 — redundancja rady: zgodność person i efektywna liczba niezależnych głosów.

Jeśli Graham, Buffett i Marks zgadzają się w 95%, płacimy trzy wywołania za jeden
głos, a „SILNY KONSENSUS” (`STRONG_CONSENSUS_THRESHOLD`, council.py) przy N_eff ≈ 1
wprowadza w błąd. Z zapisanych głosów (`council_votes`) liczymy:

  * kappa Cohena między każdą parą person i między personą a trendem LLM
    (BUY↔BULLISH, SELL↔BEARISH, HOLD↔SIDEWAYS) — zgodność ponad przypadek;
  * N_eff = N / (1 + (N − 1)·ρ̄), gdzie ρ̄ to średnia kappa par (ujemne przycięte do 0:
    niezgodność nie zwiększa liczby głosów ponad N).

Kappa bywa zdradliwa przy dominacji jednej klasy (paradoks kappa: wysoka zgodność
surowa, niska kappa), więc każda para niesie też surową zgodność, a para obu
stałych głosujących tak samo (p_e = 1) dostaje kappa 1 (pełna zgoda). Para liczy się
dopiero od `MIN_PAIR_OVERLAP` wspólnych predykcji.

Czysta domena, zero I/O.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from src.domain.council import Recommendation

# Wspólnych predykcji, od których para person ma zgodność w raporcie.
MIN_PAIR_OVERLAP = 10

# Rada poniżej tej liczby niezależnych głosów to w praktyce jeden głos: etykieta
# „silny konsensus” nie ma wtedy pokrycia (7 person z ρ̄ > 0,61 daje N_eff < 1,5).
EFFECTIVELY_SINGLE_VOTER_N_EFF = 1.5

# Nazwa „rater-a” trendu LLM w parach persona–LLM.
LLM_RATER = "LLM"

_TREND_OF_RECOMMENDATION: dict[Recommendation, str] = {
    "BUY": "BULLISH",
    "SELL": "BEARISH",
    "HOLD": "SIDEWAYS",
}


@dataclass(frozen=True)
class CouncilVoteSample:
    """Jeden zapisany głos persony w jednej predykcji."""

    prediction_id: str
    investor_name: str
    recommendation: Recommendation


@dataclass(frozen=True)
class PairAgreement:
    """Zgodność dwóch raterów na `n` wspólnych predykcjach: surowa i kappa Cohena."""

    first: str
    second: str
    n: int
    agreement: float
    kappa: float


@dataclass(frozen=True)
class CouncilRedundancy:
    """Redundancja rady na oknie zapisanych głosów.

    `pairs` — pary person od najbardziej zgodnej (największa kappa), `llm_pairs` —
    persona względem trendu LLM. `n_eff`/`mean_kappa` None = żadnej pary z dość
    dużym wspólnym oknem."""

    personas: int
    predictions: int
    pairs: tuple[PairAgreement, ...]
    llm_pairs: tuple[PairAgreement, ...]
    mean_kappa: float | None
    mean_agreement: float | None
    n_eff: float | None

    @property
    def is_effectively_single_voter(self) -> bool:
        """Rada w praktyce mówi jednym głosem (N_eff poniżej `EFFECTIVELY_SINGLE_VOTER_N_EFF`)."""
        return self.n_eff is not None and self.n_eff < EFFECTIVELY_SINGLE_VOTER_N_EFF


def raw_agreement(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Odsetek zgodnych par etykiet; None bez par."""
    if not first or len(first) != len(second):
        return None
    return sum(1 for a, b in zip(first, second, strict=True) if a == b) / len(first)


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Kappa Cohena = (p_o − p_e) / (1 − p_e); None bez par. Dwie stałe, takie same
    odpowiedzi (p_e = 1) to pełna zgoda, czyli 1."""
    observed = raw_agreement(first, second)
    if observed is None:
        return None
    n = len(first)
    first_share = Counter(first)
    second_share = Counter(second)
    expected = sum(first_share[label] * second_share[label] for label in first_share) / (n * n)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def effective_voters(personas: int, mean_kappa: float) -> float:
    """N_eff = N / (1 + (N − 1)·ρ̄), ρ̄ = średnia kappa przycięta do [0, 1]."""
    correlation = min(1.0, max(0.0, mean_kappa))
    return personas / (1.0 + (personas - 1) * correlation)


def _pair(
    first: str, second: str, a: Sequence[str], b: Sequence[str]
) -> PairAgreement | None:
    kappa = cohen_kappa(a, b)
    agreement = raw_agreement(a, b)
    if kappa is None or agreement is None:
        return None
    return PairAgreement(first=first, second=second, n=len(a), agreement=agreement, kappa=kappa)


def assess_council_redundancy(
    votes: Iterable[CouncilVoteSample],
    llm_trends: Mapping[str, str] | None = None,
    *,
    min_overlap: int = MIN_PAIR_OVERLAP,
) -> CouncilRedundancy | None:
    """Redundancja z zapisanych głosów; `llm_trends` = `{prediction_id: trend LLM}`
    dokłada zgodność persona–LLM. None, gdy żadna para person nie ma `min_overlap`
    wspólnych predykcji."""
    by_persona: dict[str, dict[str, Recommendation]] = defaultdict(dict)
    predictions: set[str] = set()
    for vote in votes:
        by_persona[vote.investor_name].setdefault(vote.prediction_id, vote.recommendation)
        predictions.add(vote.prediction_id)

    pairs: list[PairAgreement] = []
    for first, second in combinations(sorted(by_persona), 2):
        shared = sorted(by_persona[first].keys() & by_persona[second].keys())
        if len(shared) < min_overlap:
            continue
        pair = _pair(
            first,
            second,
            [by_persona[first][pid] for pid in shared],
            [by_persona[second][pid] for pid in shared],
        )
        if pair is not None:
            pairs.append(pair)
    if not pairs:
        return None
    pairs.sort(key=lambda p: (-p.kappa, -p.agreement, p.first, p.second))

    voters = {name for pair in pairs for name in (pair.first, pair.second)}
    mean_kappa = sum(pair.kappa for pair in pairs) / len(pairs)
    return CouncilRedundancy(
        personas=len(voters),
        predictions=len(predictions),
        pairs=tuple(pairs),
        llm_pairs=_llm_pairs(by_persona, llm_trends or {}, min_overlap),
        mean_kappa=mean_kappa,
        mean_agreement=sum(pair.agreement for pair in pairs) / len(pairs),
        n_eff=effective_voters(len(voters), mean_kappa),
    )


def _llm_pairs(
    by_persona: Mapping[str, Mapping[str, Recommendation]],
    llm_trends: Mapping[str, str],
    min_overlap: int,
) -> tuple[PairAgreement, ...]:
    """Zgodność każdej persony z trendem LLM na wspólnych predykcjach."""
    out: list[PairAgreement] = []
    for name in sorted(by_persona):
        shared = sorted(by_persona[name].keys() & llm_trends.keys())
        if len(shared) < min_overlap:
            continue
        pair = _pair(
            name,
            LLM_RATER,
            [_TREND_OF_RECOMMENDATION[by_persona[name][pid]] for pid in shared],
            [llm_trends[pid].strip().upper() for pid in shared],
        )
        if pair is not None:
            out.append(pair)
    out.sort(key=lambda p: (-p.kappa, p.first))
    return tuple(out)
