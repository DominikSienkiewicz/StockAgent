"""#6 — fabryka read-only tooli badawczych dla pętli tool-use LLM.

Tool-use LLM może na żądanie dociągnąć kontekst (fundamenty spółki, polskie
makro) zanim wyda werdykt. Tu definiujemy SAME narzędzia jako listę `Tool`
(JSON Schema argumentów + defensywny `func`); orkiestrator przekazuje je do
`ToolUseLLMPort.analyze_with_tools(...)`.

Kontrakt FinOps: oba toole są READ-ONLY i nie inicjują nowych płatnych
wywołań. `get_fundamentals` w Fast Loop dostaje `CachedFundamentalsAdapter`
(cache-only), a `get_macro` czyta darmowe polskie makro.

Niezmiennik odporności: KAŻDY `func` łapie wyjątki wewnętrznie i zawsze zwraca
JSON-safe dict (`{"available": ...}`). Rzucony wyjątek wewnątrz pętli
narzędziowej LLM zostałby zaprezentowany modelowi jako generyczny błąd —
zamiast tego oddajemy czysty sygnał "danych brak", a cykl leci dalej. Pola
Decimal / obiekty domenowe są mapowane na typy JSON-native (float / str /
None) — nigdy nie wyciekają jako Decimal czy dataclass.
"""

from __future__ import annotations

from typing import Any

from src.application.ports import FundamentalsPort, MacroIndicatorsPort, Tool
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.value_objects import Fundamentals


def _map_fundamentals(fundamentals: Fundamentals) -> dict[str, Any]:
    """Mapuje obiekt domenowy na JSON-safe dict (pola już float | None)."""
    return {
        "available": True,
        "trailing_pe": fundamentals.trailing_pe,
        "forward_pe": fundamentals.forward_pe,
        "peg_ratio": fundamentals.peg_ratio,
        "eps_growth_yoy": fundamentals.eps_growth_yoy,
    }


def _map_macro(snapshot: PolishMacroSnapshot) -> dict[str, Any]:
    """Mapuje snapshot makro na JSON-safe dict — Decimal → float."""
    return {
        "available": True,
        "eur_pln": float(snapshot.eur_pln),
        "usd_pln": float(snapshot.usd_pln),
        "eur_pln_30d_change_pct": float(snapshot.eur_pln_30d_change_pct),
        "usd_pln_30d_change_pct": float(snapshot.usd_pln_30d_change_pct),
    }


def build_research_tools(
    fundamentals_port: FundamentalsPort,
    macro_port: MacroIndicatorsPort,
) -> list[Tool]:
    """Buduje read-only toole badawcze dla pętli tool-use LLM.

    Zwraca dwa `Tool`-e:
    - `get_fundamentals` (wymaga `{"symbol": string}`),
    - `get_macro` (bez argumentów).

    Każdy `func` jest domknięciem nad odpowiednim portem, w pełni defensywne.
    """

    def _get_fundamentals(args: dict[str, Any]) -> dict[str, Any]:
        # Defensywnie — brak/zły symbol nie może wywalić pętli narzędziowej.
        try:
            symbol = args.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                return {"available": False}
            fundamentals = fundamentals_port.get_fundamentals(symbol)
            if fundamentals is None:
                return {"available": False}
            return _map_fundamentals(fundamentals)
        except Exception:
            # Błąd portu (np. API down) traktujemy jak brak danych.
            return {"available": False}

    def _get_macro(_args: dict[str, Any]) -> dict[str, Any]:
        try:
            snapshot = macro_port.fetch_polish_macro()
            if snapshot is None:
                return {"available": False}
            return _map_macro(snapshot)
        except Exception:
            return {"available": False}

    get_fundamentals_tool = Tool(
        name="get_fundamentals",
        description=(
            "Zwraca dane fundamentalne spółki (trailing/forward P/E, PEG, "
            "wzrost EPS r/r). Argument: symbol giełdowy. Dane mogą być "
            "niedostępne dla ETF-ów / krypto / przy braku w źródle."
        ),
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol giełdowy, np. AAPL.",
                },
            },
            "required": ["symbol"],
        },
        func=_get_fundamentals,
    )

    get_macro_tool = Tool(
        name="get_macro",
        description=(
            "Zwraca aktualne polskie makro: kurs EUR/PLN i USD/PLN oraz ich "
            "30-dniową zmianę procentową (proxy ryzyka suwerennego). "
            "Bez argumentów."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        func=_get_macro,
    )

    return [get_fundamentals_tool, get_macro_tool]
