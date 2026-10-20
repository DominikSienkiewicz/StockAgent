# src/domain/finops.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# FinOps — czysty model kosztu jednego cyklu agenta. "Agent, który raportuje
# własny rachunek": bierze liczniki płatnych wywołań zewnętrznych (LLM, sentyment,
# news, embeddingi) i przelicza je na SZACUNKOWY koszt w USD. Bez I/O, bez importów
# z application — sam rachunek na podstawie jawnego cennika.
#
# Sedno reguły FinOps repo: cykl odcięty bramką volatility nie woła żadnego
# płatnego portu, więc jego koszt to dokładnie 0.0 — i to jest najciekawsza,
# a nie najnudniejsza, informacja tej sekcji.

# ---------------------------------------------------------------------------
# CENNIK — JEDNO ŹRÓDŁO PRAWDY.
#
# UWAGA: WSZYSTKIE WARTOŚCI SĄ SZACUNKAMI RZĘDU WIELKOŚCI, NIE FAKTURĄ.
# To orientacyjny koszt jednego wywołania danego źródła, a nie rzeczywisty
# rachunek od dostawcy. Zależy od długości promptu, modelu, planu taryfowego
# i kursu — więc traktuj liczby jako "ile mniej więcej kosztuje ten cykl",
# nie jako pozycję na fakturze. Prezentowanie szacunku jako rachunku byłoby
# nieuczciwe, dlatego renderer zawsze dokleja słowo "szacunkowy".
#
# Podstawa oszacowań (stan wiedzy ~2026, publiczne cenniki):
#   - llm          — jedno zapytanie predykcyjne (model klasy gpt-5-mini /
#                    claude-sonnet), kilka tys. tokenów in+out ≈ $0.01.
#   - council_llm  — jedno wywołanie persony lub chairmana rady (gpt-5-mini);
#                    licznik liczy KAŻDY request rady (FIN-02), więc przebieg
#                    7 person + chairman to 8 wywołań, z debatą 15 ≈ $0.0025
#                    za wywołanie. Raporty sprzed FIN-02 liczyły przebieg rady
#                    jako 1 wywołanie po $0.02, czyli zaniżały radę.
#   - sentiment    — Alpha Vantage (NEWS_SENTIMENT); rozliczenie subskrypcyjne,
#                    więc koszt zamortyzowany na wywołanie jest groszowy ≈ $0.0005.
#   - news         — news API; również subskrypcja, koszt zamortyzowany ≈ $0.0005.
#   - embedding    — OpenAI text-embedding-3-small ($0.02 / 1M tokenów); zapytanie
#                    rzędu kilkuset tokenów ≈ $0.00002.
UNIT_COST_USD: dict[str, float] = {
    "llm": 0.010,
    "council_llm": 0.0025,
    "sentiment": 0.0005,
    "news": 0.0005,
    "embedding": 0.00002,
}

# Świadomie `float`, a NIE `Decimal`. Decimal ma sens w księgowości, gdzie liczy
# się grosz i zaokrąglenia prawne. Tu operujemy na SZACUNKU rzędu wielkości —
# precyzja monetarna jest iluzją, z której nikt nie wystawi faktury. `float`
# jest prostszy, wystarczająco dokładny dla kwot rzędu centów i nie udaje
# księgowej ścisłości, której ten rachunek nie ma.


@dataclass(frozen=True)
class SourceCost:
    """Jedna pozycja rozbicia: ile kosztowało dane (znane) źródło w tym cyklu."""

    source: str
    calls: int
    unit_cost_usd: float
    subtotal_usd: float


@dataclass(frozen=True)
class CycleCost:
    """Szacunkowy koszt jednego cyklu: rozbicie per źródło + suma.

    - `lines`           — rozbicie tylko po ZNANYCH źródłach (posortowane malejąco
                          po koszcie), każde z liczbą wywołań i subtotalem.
    - `total_usd`       — suma subtotali (niezmiennik: == sum(line.subtotal_usd)).
    - `total_calls`     — łączna liczba WSZYSTKICH płatnych wywołań (znane +
                          nieznane); nieznane też realnie się odbyły.
    - `unknown_sources` — źródła spoza cennika (wkład 0, ale jawnie zasygnalizowane).

    Cykl darmowy = `total_calls == 0` (bramka volatility odcięła wszystkie symbole).
    """

    lines: tuple[SourceCost, ...]
    total_usd: float
    total_calls: int
    unknown_sources: tuple[str, ...]


def estimate_cycle_cost(calls: Mapping[str, int]) -> CycleCost:
    """Szacuje koszt cyklu z liczników płatnych wywołań per źródło.

    WYNIK JEST SZACUNKIEM (patrz `UNIT_COST_USD`), nie rachunkiem. Reguły:
      - licznik <= 0 jest pomijany (nie zaniża sumy, nie tworzy pustej linii);
      - źródło spoza cennika NIE rzuca KeyError — wchodzi na listę
        `unknown_sources` z wkładem 0, ale jego wywołania liczą się do
        `total_calls` (naprawdę poszły);
      - zero wywołań → koszt 0.0 (cykl odcięty bramką volatility jest darmowy).
    """
    lines: list[SourceCost] = []
    unknown: list[str] = []
    total_calls = 0

    for source, count in calls.items():
        if count <= 0:
            continue
        total_calls += count
        unit_cost = UNIT_COST_USD.get(source)
        if unit_cost is None:
            unknown.append(source)
            continue
        subtotal = unit_cost * count
        lines.append(
            SourceCost(
                source=source,
                calls=count,
                unit_cost_usd=unit_cost,
                subtotal_usd=subtotal,
            )
        )

    # Najdroższe źródło na górze; nieznane sortujemy alfabetycznie dla determinizmu.
    lines.sort(key=lambda line: line.subtotal_usd, reverse=True)
    unknown.sort()

    return CycleCost(
        lines=tuple(lines),
        # Suma w kolejności rozbicia: dodawanie floatów nie jest łączne, więc suma
        # liczona w innej kolejności mogłaby różnić się od sumy linii o ułamek centa.
        total_usd=sum((line.subtotal_usd for line in lines), 0.0),
        total_calls=total_calls,
        unknown_sources=tuple(unknown),
    )


@dataclass(frozen=True)
class CycleCallBound:
    """Górna granica płatnych wywołań JEDNEGO cyklu (najgorszy przypadek).

    - `calls`       — per klucz `UNIT_COST_USD`, w jednostkach licznika cyklu,
                      więc `estimate_cycle_cost(calls)` to najwyższy szacunek,
                      jaki może pokazać sekcja „Koszt cyklu”.
    - `av_requests` — requesty HTTP do Alpha Vantage NEWS_SENTIMENT: limit
                      kluczy, nie pozycja cennika.
    """

    calls: Mapping[str, int]
    av_requests: int


def cycle_call_upper_bound(
    n_gated: int,
    n_av_symbols: int,
    personas: int,
    debate: bool,
    tool_rounds: int,
) -> CycleCallBound:
    """Najgorszy przypadek cyklu, w którym `n_gated` symboli przechodzi bramkę.

    Per symbol za bramką:
      - `llm`         — diagnoza pomyłki w reflect (ta sama bramka) + predykcja:
                        jedno `analyze` albo pętla tool-use, czyli `tool_rounds`
                        rund z toolami i jedna wymuszona bez nich (`tool_rounds`
                        = 0 → tool-use wyłączony);
      - `council_llm` — `personas` wywołań, drugie tyle przy debacie (runda
                        rewizji po podzielonym głosowaniu), plus chairman;
      - `sentiment`, `news` — po jednym wywołaniu węzła;
      - `embedding`   — embedding RAG w predict i ponowna próba przy zapisie,
                        gdy RAG nie dał wektora.
    Feed Alpha Vantage pobiera pierwszy symbol za bramką dla wszystkich
    `n_av_symbols` tickerów, więc `av_requests` nie zależy od liczby symboli
    za bramką. Cykl, w którym bramka odcina wszystko, ma granicę zero.

    Wynik to szacunek tak jak cennik: model routing może dołożyć eskalację do
    modelu frontier w `analyze`, a rotacja kluczy AV ponawia request po
    odpowiedzi rate-limit — ani jedno, ani drugie nie jest tu liczone.
    """
    for name, value in (
        ("n_gated", n_gated),
        ("n_av_symbols", n_av_symbols),
        ("personas", personas),
        ("tool_rounds", tool_rounds),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative (got {value})")

    prediction_requests = tool_rounds + 1 if tool_rounds > 0 else 1
    council_requests = personas * (2 if debate else 1) + 1 if personas > 0 else 0
    per_symbol = {
        "llm": 1 + prediction_requests,
        "council_llm": council_requests,
        "sentiment": 1,
        "news": 1,
        "embedding": 2,
    }
    return CycleCallBound(
        calls=MappingProxyType(
            {source: n_gated * requests for source, requests in per_symbol.items()}
        ),
        av_requests=n_av_symbols if n_gated > 0 else 0,
    )
