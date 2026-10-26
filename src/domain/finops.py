# src/domain/finops.py
from __future__ import annotations

import calendar
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from src.domain.shadow_prompt import SHADOW_PROMPT_MAX_PER_CYCLE

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
    """Jedna pozycja rozbicia: ile kosztowało dane (znane) źródło w tym cyklu.

    `from_tokens` (FIN-13): subtotal policzony z tokenów zwróconych przez dostawcę
    i cennika per model; `unit_cost_usd` to wtedy średnia na wywołanie. False =
    szacunek `UNIT_COST_USD` × liczba wywołań."""

    source: str
    calls: int
    unit_cost_usd: float
    subtotal_usd: float
    from_tokens: bool = False


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
    `priced_as_of` (FIN-13) — data cennika per model, z którego policzono linie
    `from_tokens`; None, gdy żadna linia nie jest z tokenów.
    """

    lines: tuple[SourceCost, ...]
    total_usd: float
    total_calls: int
    unknown_sources: tuple[str, ...]
    priced_as_of: date | None = None

    @property
    def from_tokens(self) -> bool:
        """Czy choć jedna linia rozbicia jest policzona z tokenów (FIN-13)."""
        return any(line.from_tokens for line in self.lines)


def estimate_cycle_cost(
    calls: Mapping[str, int],
    token_costs: Mapping[str, float] | None = None,
    priced_as_of: date | None = None,
) -> CycleCost:
    """Szacuje koszt cyklu z liczników płatnych wywołań per źródło.

    Bez `token_costs` WYNIK JEST SZACUNKIEM (patrz `UNIT_COST_USD`), nie rachunkiem.
    Reguły:
      - licznik <= 0 jest pomijany (nie zaniża sumy, nie tworzy pustej linii);
      - źródło spoza cennika NIE rzuca KeyError — wchodzi na listę
        `unknown_sources` z wkładem 0, ale jego wywołania liczą się do
        `total_calls` (naprawdę poszły);
      - zero wywołań → koszt 0.0 (cykl odcięty bramką volatility jest darmowy).

    FIN-13: `token_costs` (z `token_cost_by_source`) podmienia szacunek źródła na
    koszt z tokenów; źródło bez wpisu zostaje na `UNIT_COST_USD` (fallback per
    źródło). O tym, które źródła są w rozbiciu, decyduje wyłącznie licznik `calls`.
    """
    tokens = token_costs or {}
    lines: list[SourceCost] = []
    unknown: list[str] = []
    total_calls = 0

    for source, count in calls.items():
        if count <= 0:
            continue
        total_calls += count
        token_cost = tokens.get(source)
        if token_cost is not None:
            lines.append(
                SourceCost(
                    source=source,
                    calls=count,
                    unit_cost_usd=token_cost / count,
                    subtotal_usd=token_cost,
                    from_tokens=True,
                )
            )
            continue
        unit_cost = _unit_cost_usd(source)
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
        priced_as_of=priced_as_of if any(line.from_tokens for line in lines) else None,
    )


# ---------------------------------------------------------------------------
# FIN-13: koszt z prawdziwych tokenów.
#
# Adapter LLM / embeddingów zgłasza tokeny z odpowiedzi dostawcy (`LLMUsage`),
# a domena wycenia je cennikiem per model (`ModelPrice`, USD za 1M tokenów,
# standard tier) z `config.toml`. Model spoza cennika → None: nie zgadujemy
# stawki, źródło wraca do szacunku `UNIT_COST_USD`.

_TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class LLMUsage:
    """Tokeny JEDNEGO requestu do dostawcy — bez treści promptu i odpowiedzi.

    Kategorie są rozłączne i każda ma własną stawkę:
      - `input_tokens`       — wejście po pełnej stawce (bez odczytu i zapisu cache);
      - `output_tokens`      — wyjście (u modeli reasoning z tokenami rozumowania);
      - `cache_read_tokens`  — wejście odczytane z cache promptu;
      - `cache_write_tokens` — wejście zapisane do cache (Anthropic, zapis 5 min).
    `model` to id modelu, o który prosił adapter (klucz cennika), nie snapshot
    zwrócony przez API."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative (got {value})")


@dataclass(frozen=True)
class ModelPrice:
    """Cennik jednego modelu: USD za 1M tokenów danej kategorii.

    None = cennik nie podaje stawki tej kategorii (OpenAI nie nalicza zapisu cache,
    embedding nie ma wyjścia). Request z tokenami takiej kategorii nie ma wyceny."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float | None = None
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None


def price_usage(usage: LLMUsage, price_table: Mapping[str, ModelPrice]) -> float | None:
    """Koszt requestu w USD z tokenów i cennika per model.

    None, gdy modelu nie ma w cenniku albo request ma tokeny kategorii bez stawki —
    wtedy koszt jest nieznany, a nie zerowy."""
    price = price_table.get(usage.model)
    if price is None:
        return None
    total = 0.0
    for tokens, rate in (
        (usage.input_tokens, price.input_usd_per_mtok),
        (usage.output_tokens, price.output_usd_per_mtok),
        (usage.cache_read_tokens, price.cache_read_usd_per_mtok),
        (usage.cache_write_tokens, price.cache_write_usd_per_mtok),
    ):
        if tokens == 0:
            continue
        if rate is None:
            return None
        total += tokens * rate / _TOKENS_PER_PRICE_UNIT
    return total


def token_cost_by_source(
    records: Iterable[tuple[str, LLMUsage]], price_table: Mapping[str, ModelPrice]
) -> dict[str, float]:
    """Koszt z tokenów per klucz cennika cyklu (`llm`, `council_llm`, `embedding`).

    Źródło, w którym choć jeden request nie ma wyceny, wypada z wyniku w całości:
    częściowa suma zaniżałaby je pod etykietą „z tokenów”, więc sekcja kosztu
    wraca dla niego do szacunku `UNIT_COST_USD`."""
    costs: dict[str, float] = {}
    unpriced: set[str] = set()
    for source, usage in records:
        cost = price_usage(usage, price_table)
        if cost is None:
            unpriced.add(source)
            continue
        costs[source] = costs.get(source, 0.0) + cost
    return {source: cost for source, cost in costs.items() if source not in unpriced}


@dataclass(frozen=True)
class LLMCallEntry:
    """Wiersz ledgera `llm_calls` (migracja 040): jeden request do dostawcy.

    Same liczby i identyfikatory przebiegu — nigdy treść promptu ani odpowiedzi
    (prompt niesie kontekst portfela). `cost_usd` None = model spoza cennika;
    `price_as_of` = data cennika, którym policzono `cost_usd`."""

    run_id: str
    node: str
    symbol: str | None
    usage: LLMUsage
    cost_usd: float | None
    price_as_of: date | None


@dataclass(frozen=True)
class LedgerCost:
    """FIN-26: odczyt wiersza ledgera `llm_calls` do kosztu tygodnia — symbol (None
    dla jobów Slow Loopa) i koszt z tokenów (None = model spoza cennika)."""

    symbol: str | None
    cost_usd: float | None


@dataclass(frozen=True)
class CycleCallBound:
    """Górna granica płatnych wywołań JEDNEGO cyklu (najgorszy przypadek).

    - `calls`       — per klucz `UNIT_COST_USD`, w jednostkach licznika cyklu,
                      więc `estimate_cycle_cost(calls)` to najwyższy szacunek,
                      jaki może pokazać sekcja „Koszt cyklu”.
    - `av_requests` — requesty HTTP do Alpha Vantage NEWS_SENTIMENT: limit
                      kluczy, nie pozycja cennika. Po jednym na symbol za
                      bramką, który AV przyjmuje (klient leniwy, FIN-04).
    """

    calls: Mapping[str, int]
    av_requests: int


# Diagnoza pomyłki w reflect: jeden request `llm` na symbol za bramką, przed sufitem FIN-12.
_DIAGNOSIS_REQUESTS = 1
# FIN-21: predykcja bez tool-use to jedno `analyze` kaskady routingu (pętla tool-use
# idzie własnym adapterem), więc na symbol eskalować może najwyżej jedna predykcja.
_CASCADE_PREDICTION_ANSWERS = 1


def cycle_call_upper_bound(
    n_gated: int,
    n_av_symbols: int,
    personas: int,
    debate: bool,
    tool_rounds: int,
    *,
    max_escalations: int = 0,
    max_shadow_calls: int = 0,
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
    Fast Loop pyta Alpha Vantage leniwie per symbol (FIN-04): request idzie
    tylko dla symbolu za bramką, więc `av_requests` = min(`n_gated`,
    `n_av_symbols`) — ani więcej niż symboli za bramką, ani więcej niż
    tickerów, które AV przyjmuje. Cykl, w którym bramka odcina wszystko, ma
    granicę zero.

    Na cykl (WIA-11), oba 0 = granica sprzed flag:
      - `llm` += eskalacje kaskady do frontiera (FIN-21): najwyżej
                        `max_escalations` (`routing_max_escalations`) i nie więcej
                        niż odpowiedzi taniego modelu, które mogą eskalować —
                        predykcja i każdy request rady, gdy rada dzieli główny LLM;
      - `shadow_llm`  — wywołania wariantu w cieniu (EXP-26): jedno na symbol za
                        bramką, najwyżej `max_shadow_calls`, nigdy ponad
                        `SHADOW_PROMPT_MAX_PER_CYCLE`; klucz tylko, gdy > 0.

    Wynik to szacunek tak jak cennik: rotacja kluczy AV ponawia request po
    odpowiedzi rate-limit i tego tu nie liczymy.
    """
    for name, value in (
        ("n_gated", n_gated),
        ("n_av_symbols", n_av_symbols),
        ("personas", personas),
        ("tool_rounds", tool_rounds),
        ("max_escalations", max_escalations),
        ("max_shadow_calls", max_shadow_calls),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative (got {value})")

    prediction_requests = tool_rounds + 1 if tool_rounds > 0 else 1
    council_requests = personas * (2 if debate else 1) + 1 if personas > 0 else 0
    per_symbol = {
        "llm": _DIAGNOSIS_REQUESTS + prediction_requests,
        "council_llm": council_requests,
        "sentiment": 1,
        "news": 1,
        "embedding": 2,
    }
    calls = {source: n_gated * requests for source, requests in per_symbol.items()}
    escalating_answers = n_gated * (_CASCADE_PREDICTION_ANSWERS + council_requests)
    calls["llm"] += min(max_escalations, escalating_answers)
    shadow_calls = min(max_shadow_calls, SHADOW_PROMPT_MAX_PER_CYCLE, n_gated)
    if shadow_calls > 0:
        calls[SHADOW_LLM_SOURCE] = shadow_calls
    return CycleCallBound(
        calls=MappingProxyType(calls),
        av_requests=min(n_gated, n_av_symbols),
    )


def capped_cycle_call_upper_bound(
    n_gated: int,
    n_av_symbols: int,
    personas: int,
    debate: bool,
    tool_rounds: int,
    cap: int | None,
    concurrency: int,
    *,
    max_escalations: int = 0,
    max_shadow_calls: int = 0,
) -> CycleCallBound:
    """WIA-11: `cycle_call_upper_bound` przy miękkim suficie FIN-12 (`cap`).

    Symbol wchodzi do analizy, dopóki licznik `llm` + `council_llm` < `cap`, a po
    ostatnim wpuszczonym kończy analizę najwyżej `concurrency` symboli w locie.
    Stąd dwie granice:
      - analizowanych symboli ≤ (`cap` − 1) + `concurrency` — każdy dokłada do
        licznika co najmniej swoją predykcję (rada odpada poniżej
        `council_volatility_threshold`), więc przy krótkiej watchliście sufit nie
        tnie predykcji, sentymentu, newsów ani embeddingów;
      - requestów rady ≤ (`cap` − 1) + `concurrency` × requesty rady na symbol.
    Diagnoza pomyłki w reflect idzie przed sufitem, więc płaci ją także symbol
    odcięty budżetem. `cap` None = granica bez sufitu. Każde źródło dostaje
    osobno swoje maksimum, więc suma to górna granica, nie scenariusz.
    Eskalacje frontiera (FIN-21) liczą się do licznika, więc tylko zmniejszają
    liczbę wpuszczonych symboli; wariant w cieniu (EXP-26) idzie wyłącznie w luzie
    sufitu (`shadow_call_allowed`), więc nigdy nie ma go więcej niż `cap`.
    """
    extras = {"max_escalations": max_escalations, "max_shadow_calls": max_shadow_calls}
    bound = cycle_call_upper_bound(n_gated, n_av_symbols, personas, debate, tool_rounds, **extras)
    if cap is None:
        return bound
    for name, value in (("cap", cap), ("concurrency", concurrency)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1 (got {value})")

    analysed = min(n_gated, cap - 1 + concurrency)
    council_per_symbol = cycle_call_upper_bound(1, 1, personas, debate, tool_rounds).calls[
        "council_llm"
    ]
    analysed_bound = cycle_call_upper_bound(
        analysed, n_av_symbols, personas, debate, tool_rounds, **extras
    )
    calls = dict(analysed_bound.calls)
    calls["llm"] += (n_gated - analysed) * _DIAGNOSIS_REQUESTS
    calls["council_llm"] = min(calls["council_llm"], cap - 1 + concurrency * council_per_symbol)
    if SHADOW_LLM_SOURCE in calls:
        calls[SHADOW_LLM_SOURCE] = min(calls[SHADOW_LLM_SOURCE], cap)
    return CycleCallBound(calls=MappingProxyType(calls), av_requests=analysed_bound.av_requests)


# WIA-11: cron Fast Loopa chodzi w dni robocze (`1-5`), więc tydzień ma 5 dni cykli.
TRADING_DAYS_PER_WEEK = 5


@dataclass(frozen=True)
class WeeklyCostBudget:
    """Tygodniowy najgorszy przypadek Fast Loopa vs jawny budżet z config.toml.

    - `cycle_worst_usd` — szacunek najdroższego cyklu (`estimate_cycle_cost`
                          granicy przy suficie FIN-12);
    - `cycles_per_week` — cykle na dobę × `TRADING_DAYS_PER_WEEK`;
    - `budget_usd`      — `fast_loop_weekly_budget_usd`; None = bez kontroli.
    """

    cycle_worst_usd: float
    cycles_per_week: int
    budget_usd: float | None

    @property
    def worst_case_usd(self) -> float:
        return self.cycle_worst_usd * self.cycles_per_week

    @property
    def ok(self) -> bool:
        return self.budget_usd is None or self.worst_case_usd <= self.budget_usd


def weekly_cost_budget(
    bound: CycleCallBound, cycles_per_day: int, budget_usd: float | None
) -> WeeklyCostBudget:
    """Najgorszy tydzień = koszt cyklu z `bound` × cykle na dobę × 5 dni handlowych.

    Szacunek jak cennik `UNIT_COST_USD`. Ręczny dispatch ponad harmonogram to cykl
    spoza tego rachunku, jak w budżecie AV."""
    if cycles_per_day < 1:
        raise ValueError(f"cycles_per_day must be >= 1 (got {cycles_per_day})")
    if budget_usd is not None and budget_usd < 0:
        raise ValueError(f"budget_usd must be non-negative or unset (got {budget_usd})")
    return WeeklyCostBudget(
        cycle_worst_usd=estimate_cycle_cost(bound.calls).total_usd,
        cycles_per_week=cycles_per_day * TRADING_DAYS_PER_WEEK,
        budget_usd=budget_usd,
    )


# FIN-32: darmowy klucz Alpha Vantage daje 25 zapytań na dobę (reset o północy UTC).
AV_DAILY_REQUESTS_PER_KEY = 25


@dataclass(frozen=True)
class AvBudget:
    """Dobowy budżet Alpha Vantage: najgorszy przypadek vs pojemność kluczy.

    - `worst_case` — zapytania doby, gdy każdy symbol AV przechodzi bramkę
                     w każdym cyklu;
    - `capacity`   — `AV_DAILY_REQUESTS_PER_KEY` × liczba kluczy.
    """

    worst_case: int
    capacity: int

    @property
    def ok(self) -> bool:
        return self.worst_case <= self.capacity


def av_daily_budget(
    n_av_symbols: int,
    requests_per_symbol: int,
    cycles_per_day: int,
    n_keys: int,
    *,
    requests_per_cycle: int = 0,
) -> AvBudget:
    """Niezmiennik: (symbole × zapytania na symbol + zapytania na cykl) × cykle doby
    ≤ 25 × klucze. `requests_per_cycle` (FIN-09) to zapytania niezależne od liczby
    symboli — kalendarz wyników pobierany raz na cykl.

    Bez niego przekroczenie wychodzi dopiero jako `av_keys_exhausted` w środku
    cyklu, gdy część symboli dostała już pusty feed — a za bramką i tak płaci
    za LLM. Rotacja kluczy po odpowiedzi rate-limit dokłada requesty spoza
    tego rachunku (jak w `cycle_call_upper_bound`)."""
    for name, value in (
        ("n_av_symbols", n_av_symbols),
        ("requests_per_symbol", requests_per_symbol),
        ("cycles_per_day", cycles_per_day),
        ("n_keys", n_keys),
        ("requests_per_cycle", requests_per_cycle),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative (got {value})")
    return AvBudget(
        worst_case=(n_av_symbols * requests_per_symbol + requests_per_cycle) * cycles_per_day,
        capacity=AV_DAILY_REQUESTS_PER_KEY * n_keys,
    )


# FIN-12: źródła cennika liczone do sufitu płatnych wywołań LLM cyklu — predykcja
# z diagnozą pomyłki (`llm`) i każdy request rady (`council_llm`). Sentyment, newsy
# i embeddingi mają własne limity (klucze AV, grosze za embedding), nie ten sufit.
PAID_LLM_SOURCES: frozenset[str] = frozenset({"llm", "council_llm"})


def paid_llm_calls(calls: Mapping[str, int]) -> int:
    """Wywołania LLM cyklu (`PAID_LLM_SOURCES`) z liczników per klucz cennika."""
    return sum(count for source, count in calls.items() if source in PAID_LLM_SOURCES)


def paid_budget_allows(spent_calls: int, cap: int | None) -> bool:
    """Czy kolejny symbol za bramką volatility może jeszcze płacić za LLM.

    Miękki sufit: wpuszcza, dopóki wydane < `cap`, więc ostatni wpuszczony symbol
    może go przebić o swoje wywołania (predykcja + rada) — nie przerywamy analizy
    w połowie. `cap` None = bez limitu."""
    if spent_calls < 0:
        raise ValueError(f"spent_calls must be non-negative (got {spent_calls})")
    return cap is None or spent_calls < cap


def budget_exceeded(spent_usd: float, cap_usd: float | None) -> bool:
    """FIN-25: czy limit USD okna (doba, miesiąc) jest wyczerpany.

    Wyczerpany już przy wydanych == limit: sprawdzenie idzie raz przed cyklem, więc
    wpuszczony cykl przebiłby limit o swój koszt. Ten sam powód sprawia, że limit
    jest twardy dopiero od następnego cyklu — ostatni wpuszczony może go przekroczyć
    o własny koszt. `cap_usd` None = bez limitu, 0 = okno bez płatnych wywołań."""
    if spent_usd < 0:
        raise ValueError(f"spent_usd must be non-negative (got {spent_usd})")
    return cap_usd is not None and spent_usd >= cap_usd


class BudgetWindow(StrEnum):
    """FIN-25: okno limitu USD — doba albo miesiąc kalendarzowy UTC."""

    DAY = "day"
    MONTH = "month"


@dataclass(frozen=True)
class BudgetBreach:
    """FIN-25: wyczerpany limit okna — ile wydano z ledgera i jaki był limit."""

    window: BudgetWindow
    spent_usd: float
    cap_usd: float


def budget_window_start(window: BudgetWindow, now: datetime) -> datetime:
    """Początek okna w UTC: północ doby albo pierwszy dzień miesiąca.

    UTC, bo w nim liczy się `llm_calls.created_at` i reset dostawców — strefa maila
    przesuwałaby granicę doby o godzinę-dwie zależnie od czasu letniego."""
    if now.tzinfo is None:
        raise ValueError("budget window needs a timezone-aware clock")
    day = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return day if window is BudgetWindow.DAY else day.replace(day=1)


@dataclass(frozen=True)
class MonthToDateCost:
    """FIN-23: koszt miesiąca kalendarzowego UTC narastająco z ledgera `llm_calls`.

    `days_elapsed` liczy dzień bieżący w całości (1. dnia miesiąca = 1), więc
    prognoza nie wybucha w pierwszych godzinach miesiąca."""

    month_start: date
    spent_usd: float
    days_elapsed: int
    days_in_month: int

    @property
    def projected_usd(self) -> float:
        """Prognoza liniowa do końca miesiąca: średnia dzienna × dni miesiąca.
        Szacunek, nie faktura — tempo pierwszych dni nie musi się utrzymać."""
        return self.spent_usd / self.days_elapsed * self.days_in_month


def month_to_date_cost(spent_usd: float, now: datetime) -> MonthToDateCost:
    """FIN-23: wydatek od początku miesiąca UTC (to samo okno co limit miesięczny
    FIN-25, `budget_window_start`) i dni, z których prognoza liczy tempo."""
    if spent_usd < 0:
        raise ValueError(f"spent_usd must be non-negative (got {spent_usd})")
    start = budget_window_start(BudgetWindow.MONTH, now)
    today = now.astimezone(UTC)
    return MonthToDateCost(
        month_start=start.date(),
        spent_usd=spent_usd,
        days_elapsed=today.day,
        days_in_month=calendar.monthrange(today.year, today.month)[1],
    )


# EXP-26: klucz cennika wywołań wariantu promptu albo modelu w cieniu. Poza
# `PAID_LLM_SOURCES`, więc sufit produkcji (`paid_budget_allows`) go nie widzi.
SHADOW_LLM_SOURCE = "shadow_llm"

# WIA-11: stałe źródeł eksperymentów za flagą. Wariant w cieniu to prompt albo model
# głównego LLM, więc kosztuje jak predykcja. Poza `UNIT_COST_USD`, bo cykl bez flagi
# nie ma tego źródła — tabela koperty FIN-06 i jej kolejność zostają jak przed EXP-26.
EXPERIMENT_UNIT_COST_USD: dict[str, float] = {SHADOW_LLM_SOURCE: UNIT_COST_USD["llm"]}


def _unit_cost_usd(source: str) -> float | None:
    """Stała źródła z cennika produkcji albo eksperymentów; None = spoza cennika."""
    unit_cost = UNIT_COST_USD.get(source)
    return EXPERIMENT_UNIT_COST_USD.get(source) if unit_cost is None else unit_cost


def shadow_call_allowed(
    calls: Mapping[str, int], *, cap: int | None, max_shadow_calls: int
) -> bool:
    """EXP-26: czy wariant promptu albo modelu w cieniu dostanie kolejne wywołanie LLM.

    Wywołanie cienia liczy się do sufitu cyklu, i to twardo: idzie tylko wtedy, gdy
    WSZYSTKIE wywołania LLM cyklu (produkcja i cień) razem z nim mieszczą się w `cap`.
    Produkcja liczy swój miękki sufit bez cienia (`PAID_LLM_SOURCES`), więc cień nie
    zabiera jej przydziału — dostaje wyłącznie luz, który został w chwili wywołania.
    Liczbę wywołań cienia ogranicza `max_shadow_calls`, ale nigdy ponad
    `SHADOW_PROMPT_MAX_PER_CYCLE`. `cap` None = bez sufitu cyklu."""
    shadow_spent = calls.get(SHADOW_LLM_SOURCE, 0)
    if shadow_spent >= min(max_shadow_calls, SHADOW_PROMPT_MAX_PER_CYCLE):
        return False
    return cap is None or paid_llm_calls(calls) + shadow_spent + 1 <= cap


def rank_by_magnitude(deltas: Mapping[str, Decimal]) -> list[str]:
    """Symbole malejąco po |Δ|; remis rozstrzyga alfabet.

    Kolejność alfabetyczna (albo z config.toml) systematycznie odcinałaby budżetem
    te same symbole — największe ruchy mają pierwszeństwo do płatnej analizy."""
    return sorted(deltas, key=lambda symbol: (-abs(deltas[symbol]), symbol))


def slow_loop_paid_budget(calls_by_job: Mapping[str, int], cap: int | None) -> dict[str, int]:
    """FIN-30: przydział tygodniowego budżetu płatnych wywołań LLM Slow Loopa.

    Slow Loop chodzi poza bramką volatility (recap, sędzia kalibracji, przyszłe
    joby diagnoz), więc jego koszt ogranicza ten sufit, a nie ruch ceny. Kolejność
    `calls_by_job` to pierwszeństwo: każdy job dostaje min(żądane, reszta budżetu),
    więc ostatni jest przycinany pierwszy. `cap` None = każdy job dostaje wszystko."""
    if cap is not None and cap < 0:
        raise ValueError(f"cap must be non-negative (got {cap})")
    for job, requested in calls_by_job.items():
        if requested < 0:
            raise ValueError(f"{job} must request a non-negative number of calls (got {requested})")
    if cap is None:
        return dict(calls_by_job)
    remaining = cap
    allowed: dict[str, int] = {}
    for job, requested in calls_by_job.items():
        allowed[job] = min(requested, remaining)
        remaining -= allowed[job]
    return allowed
