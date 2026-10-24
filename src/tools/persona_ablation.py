"""WIA-46 — prequential ablacja ważenia person rady (`persona_weighting_enabled`).

Pytanie: czy werdykt rady z wagami person jest trafniejszy niż bez nich? RPC 015
liczy wagi z okna od NOW(), więc retrospektywna odpowiedź użyłaby wag, które znają
wynik ocenianej predykcji, i zawyżyłaby zysk z ważenia. Tu każda historyczna
predykcja dostaje wagi z chwili, w której powstała: wyłącznie z głosów
rozliczonych wcześniej (`persona_accuracy_weights_as_of`, rozliczenie bez stempla
przybliżone horyzontem prognozy). Werdykt liczy ta sama reguła co Fast Loop
(`derive_consensus`), a jego trafność — reguła głosu persony
(`persona_vote_correct`, pasmo HOLD wg klasy aktywa).

Wynik to para werdyktów na każdej predykcji: trafienia z wagami i bez nich oraz
test znaków na predykcjach, gdzie trafiła dokładnie jedna wersja (dokładny
McNemar, jednostronnie: ważenie lepsze). Dopiero przewaga poza próbą uzasadnia
włączenie `persona_weighting_enabled`.

Wejście: surowe głosy z RPC `persona_vote_outcomes` (migracja 045). Złożenie
`Settings` i repozytorium robi `main_tools.py persona-ablation` (ARCH-39);
read-only, zero płatnych wywołań.

Koszt: wagi każdej predykcji ocenia od nowa domena na wszystkich głosach
(O(predykcje × głosy)) — przy oknie 90 dni i radzie, która głosuje tylko ponad
`council_volatility_threshold`, to sekundy; jedna reguła ważniejsza niż czas.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from src.domain.council import InvestorOpinion, derive_consensus, parse_recommendation
from src.domain.persona_track_record import (
    PersonaVoteOutcome,
    persona_accuracy_weights_as_of,
    persona_vote_correct,
)
from src.domain.significance import binomial_test_greater
from src.domain.value_objects import AssetType

# Poziom testu znaków, od którego wynik uzasadnia włączenie ważenia.
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class PersonaAblation:
    """Werdykty rady z wagami as-of i bez wag na tych samych predykcjach.

    `without_weights` — predykcje bez żadnego wcześniej rozliczonego głosu (oba
    werdykty identyczne). `weighted_only` / `unweighted_only` — trafiła tylko
    ta wersja. `p_value` — P(co najmniej `weighted_only` z par rozbieżnych przy
    braku różnicy); None bez par rozbieżnych."""

    evaluated: int
    without_weights: int
    weighted_hits: int
    unweighted_hits: int
    weighted_only: int
    unweighted_only: int
    p_value: float | None


def _opinions(votes: Iterable[PersonaVoteOutcome]) -> list[InvestorOpinion]:
    """Głosy predykcji jako opinie rady; rekomendacja spoza BUY/SELL/HOLD odpada."""
    opinions: list[InvestorOpinion] = []
    for vote in votes:
        recommendation = parse_recommendation(vote.recommendation)
        if recommendation is not None:
            opinions.append(
                InvestorOpinion(vote.investor_name, recommendation, vote.confidence, "", ())
            )
    return opinions


def _verdict_hit(
    recommendation: str, sample: PersonaVoteOutcome, crypto: Collection[str]
) -> bool:
    """Trafność werdyktu rady na cenach predykcji — reguła głosu persony."""
    asset_type = AssetType.CRYPTO if sample.symbol.upper() in crypto else None
    return persona_vote_correct(
        recommendation, sample.price_at_prediction, sample.actual_price, asset_type
    )


def replay_persona_weighting(
    outcomes: Sequence[PersonaVoteOutcome],
    *,
    since: datetime | None = None,
    window_days: int = 90,
    crypto_symbols: Collection[str] = (),
) -> PersonaAblation:
    """Odtwarza werdykt rady każdej predykcji od `since` (None = wszystkie)
    z wagami prequential i bez wag. Historia sprzed `since` zasila tylko wagi."""
    crypto = frozenset(symbol.upper() for symbol in crypto_symbols)
    by_prediction: dict[str, list[PersonaVoteOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.predicted_at is not None and (since is None or outcome.predicted_at >= since):
            by_prediction[outcome.prediction_id].append(outcome)

    evaluated = without_weights = weighted_hits = unweighted_hits = 0
    weighted_only = unweighted_only = 0
    for votes in by_prediction.values():
        opinions = _opinions(votes)
        sample = votes[0]
        if not opinions or sample.predicted_at is None:
            continue
        weights = persona_accuracy_weights_as_of(
            outcomes, as_of=sample.predicted_at, window_days=window_days, crypto_symbols=crypto
        )
        weighted = _verdict_hit(derive_consensus(opinions, weights)[0], sample, crypto)
        unweighted = _verdict_hit(derive_consensus(opinions)[0], sample, crypto)
        evaluated += 1
        without_weights += not weights
        weighted_hits += weighted
        unweighted_hits += unweighted
        weighted_only += weighted and not unweighted
        unweighted_only += unweighted and not weighted

    discordant = weighted_only + unweighted_only
    return PersonaAblation(
        evaluated=evaluated,
        without_weights=without_weights,
        weighted_hits=weighted_hits,
        unweighted_hits=unweighted_hits,
        weighted_only=weighted_only,
        unweighted_only=unweighted_only,
        p_value=binomial_test_greater(weighted_only, discordant, 0.5) if discordant else None,
    )


def _rate(hits: int, total: int) -> str:
    return f"{hits / total * 100:.1f}%" if total else "—"


def format_ablation(report: PersonaAblation, *, days: int, window_days: int) -> str:
    """Raport CLI ablacji; ostatnia linia mówi, czy wynik uzasadnia flagę."""
    n = report.evaluated
    lines = [
        f"Ablacja ważenia person (prequential, ostatnie {days} dni, okno wag {window_days} dni)",
        f"  Predykcje z głosami rady:       n={n} "
        f"(bez wcześniejszych wag: {report.without_weights})",
        f"  Trafność werdyktu z wagami:     {_rate(report.weighted_hits, n)}",
        f"  Trafność werdyktu bez wag:      {_rate(report.unweighted_hits, n)}",
        "  Rozbieżne (trafiła jedna wersja): "
        f"z wagami {report.weighted_only}, bez wag {report.unweighted_only}",
    ]
    if report.p_value is None:
        lines.append("  Test znaków:                    — (brak rozbieżnych werdyktów)")
    else:
        lines.append(f"  Test znaków (ważenie lepsze):   p={report.p_value:.3f}")
    if report.p_value is not None and report.p_value < SIGNIFICANCE_LEVEL:
        lines.append(
            "  Werdykt: ważenie trafniejsze poza próbą — wynik uzasadnia "
            "persona_weighting_enabled = true."
        )
    else:
        lines.append(
            "  Werdykt: brak dowodu, że ważenie pomaga poza próbą — zostaw "
            "persona_weighting_enabled = false."
        )
    return "\n".join(lines)
