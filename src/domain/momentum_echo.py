"""Test echa momentum (EXP-01): czy trend LLM to tylko znak ruchu, który otworzył bramkę.

Czysta domena (stdlib + inne moduły domeny). Bramka volatility przepuszcza
tylko |Δ| ≥ próg, a LLM widzi tę samą zmianę w prompcie. Jeśli trend w większości
powtarza jej znak (echo momentum) albo go odwraca, darmowa reguła daje to samo
za 0 zł — to najtańszy falsyfikator wartości płatnej nogi. Liczymy:

  (a) odsetek echa (`predicted_trend == sign(Δ bramki)`) i odwrócenia
      (`== −sign`) wśród predykcji z deltą bramki,
  (b) hit-rate darmowych baseline'ów „znak ruchu” i „przeciwny znak” na tych
      samych wierszach co hit-rate LLM (zakład kierunkowy LLM, Δ ≠ 0, znany ruch),
  (c) hit-rate LLM wyłącznie tam, gdzie nie zgadza się z momentum.

Porównanie LLM z baseline'em jest SPAROWANE: tam, gdzie obaj stawiają to samo,
trafiają i pudłują razem, więc o przewadze decydują wyłącznie wiersze rozbieżne.
Na nich (przy ruchu ≠ 0) trafienie jednego jest pudłem drugiego, więc to test
znaków: trafienia LLM względem 0.5, na n_eff po cyklach (predykcje jednego dnia
ruszają się razem). LLM vs „znak ruchu” rozstrzygają odwrócenia, LLM vs
„przeciwny znak” — echa. Obie hipotezy koryguje BH-FDR.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from src.domain.equity_curve import signal_of
from src.domain.significance import (
    DEFAULT_FDR_Q,
    benjamini_hochberg,
    binomial_test_greater,
    effective_n,
)
from src.domain.track_record import group_by_cycle

# H0 testu znaków: na wierszu rozbieżnym LLM i baseline trafiają równie często.
_SIGN_TEST_P0 = 0.5


@dataclass(frozen=True)
class EchoCall:
    """Rozliczona predykcja z deltą bramki jako prymitywy domeny.

    `gate_delta` — ułamkowa zmiana ceny, która otworzyła bramkę;
    `realized_change` — zrealizowany ruch po predykcji (None = brak cen).
    """

    cycle: str
    predicted_trend: str
    gate_delta: float
    realized_change: float | None


@dataclass(frozen=True)
class HeadToHead:
    """LLM vs darmowy baseline na wierszach rozbieżnych z ruchem ≠ 0 (test znaków).

    `wins` — trafienia LLM, czyli pudła baseline'u; `p_value` — P(X ≥ k) dla
    Binomial(n_eff, 0.5), None bez wierszy rozbieżnych; `fdr_significant` — po
    BH-FDR w rodzinie porównań testu echa.
    """

    wins: int
    n: int
    n_eff: float
    p_value: float | None
    fdr_significant: bool

    @property
    def rate(self) -> float | None:
        return _ratio(self.wins, self.n)


@dataclass(frozen=True)
class EchoReport:
    """Wynik testu echa. `with_delta` to mianownik (a), `compared` — (b),
    `disagreement` — (c); `n_eff` liczone po cyklach na wierszach (b)."""

    with_delta: int
    echo: int
    reversal: int
    compared: int
    n_eff: float
    llm_hits: int
    momentum_hits: int
    contrarian_hits: int
    disagreement: int
    disagreement_hits: int
    vs_momentum: HeadToHead
    vs_contrarian: HeadToHead

    @property
    def echo_rate(self) -> float | None:
        return _ratio(self.echo, self.with_delta)

    @property
    def reversal_rate(self) -> float | None:
        return _ratio(self.reversal, self.with_delta)

    @property
    def llm_rate(self) -> float | None:
        return _ratio(self.llm_hits, self.compared)

    @property
    def momentum_rate(self) -> float | None:
        return _ratio(self.momentum_hits, self.compared)

    @property
    def contrarian_rate(self) -> float | None:
        return _ratio(self.contrarian_hits, self.compared)

    @property
    def disagreement_rate(self) -> float | None:
        return _ratio(self.disagreement_hits, self.disagreement)


def assess_momentum_echo(calls: Sequence[EchoCall], *, q: float = DEFAULT_FDR_Q) -> EchoReport:
    """Liczy (a), (b), (c) i sparowane porównania LLM z oboma baseline'ami.

    Trafienie kierunku jak w `Prediction.is_trend_correct`: zakład w górę
    wymaga ruchu > 0, w dół < 0, więc ruch równy 0 to pudło obu stron i remis
    poza testem znaków.
    """
    compared = [c for c in calls if _stance(c) != 0 and c.realized_change is not None]
    agreeing = [c for c in compared if _stance(c) == 1]
    disagreeing = [c for c in compared if _stance(c) == -1]

    # Kolejność rodziny BH: (LLM vs „znak ruchu”, LLM vs „przeciwny znak”).
    contests = (_sign_test(disagreeing), _sign_test(agreeing))
    tested = [(i, h.p_value) for i, h in enumerate(contests) if h.p_value is not None]
    rejected = benjamini_hochberg([p for _, p in tested], q)
    significant = {i for (i, _), flag in zip(tested, rejected, strict=True) if flag}
    vs_momentum, vs_contrarian = (
        dataclasses.replace(h, fdr_significant=i in significant) for i, h in enumerate(contests)
    )

    return EchoReport(
        with_delta=len(calls),
        echo=sum(1 for c in calls if _stance(c) == 1),
        reversal=sum(1 for c in calls if _stance(c) == -1),
        compared=len(compared),
        n_eff=_n_eff(compared),
        llm_hits=sum(1 for c in compared if _llm_hit(c)),
        momentum_hits=sum(1 for c in compared if _hit(_momentum(c), c)),
        contrarian_hits=sum(1 for c in compared if _hit(-_momentum(c), c)),
        disagreement=len(disagreeing),
        disagreement_hits=sum(1 for c in disagreeing if _llm_hit(c)),
        vs_momentum=vs_momentum,
        vs_contrarian=vs_contrarian,
    )


def _momentum(call: EchoCall) -> int:
    if call.gate_delta > 0:
        return 1
    if call.gate_delta < 0:
        return -1
    return 0


def _stance(call: EchoCall) -> int:
    """1 = echo momentum, −1 = odwrócenie, 0 = brak zakładu LLM albo Δ = 0."""
    return signal_of(call.predicted_trend) * _momentum(call)


def _hit(signal: int, call: EchoCall) -> bool:
    return call.realized_change is not None and signal * call.realized_change > 0


def _llm_hit(call: EchoCall) -> bool:
    return _hit(signal_of(call.predicted_trend), call)


def _n_eff(calls: Sequence[EchoCall]) -> float:
    return effective_n(group_by_cycle((c.cycle, _llm_hit(c)) for c in calls))


def _sign_test(disputed: Sequence[EchoCall]) -> HeadToHead:
    """Test znaków na wierszach, gdzie LLM i baseline stawiają przeciwnie.

    Remisy (ruch 0 — obaj pudłują) wypadają. Liczności skalujemy do n_eff jak
    w `track_record._p_value_vs_best_baseline`."""
    decisive = [c for c in disputed if c.realized_change != 0]
    wins = sum(1 for c in decisive if _llm_hit(c))
    n_eff = _n_eff(decisive)
    p_value = None
    if decisive:
        scaled_n = max(1, round(min(n_eff, len(decisive))))
        p_value = binomial_test_greater(
            round(wins / len(decisive) * scaled_n), scaled_n, _SIGN_TEST_P0
        )
    return HeadToHead(
        wins=wins, n=len(decisive), n_eff=n_eff, p_value=p_value, fdr_significant=False
    )


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None
