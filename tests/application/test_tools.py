"""Testy fabryki read-only tooli badawczych (#6).

`build_research_tools` produkuje listę `Tool` dla pętli tool-use LLM:
- `get_fundamentals` — woła `FundamentalsPort.get_fundamentals(symbol)`,
- `get_macro` — woła `MacroIndicatorsPort.fetch_polish_macro()`.

Każdy `func` MUSI być defensywny: łapie wyjątki wewnętrznie i zwraca JSON-safe
dict, bo działa w pętli narzędziowej LLM (rzucony wyjątek = generyczny błąd
w SDK function-calling). Pola Decimal / obiekty domenowe są mapowane na typy
JSON-native (float / str / None), nigdy nie wyciekają jako Decimal/dataclass.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import FundamentalsPort, MacroIndicatorsPort, Tool
from src.application.tools import build_research_tools
from src.domain.polish_macro import PolishMacroSnapshot
from src.domain.value_objects import Fundamentals


def _make_fundamentals() -> Fundamentals:
    return Fundamentals(
        trailing_pe=28.5,
        forward_pe=24.0,
        peg_ratio=1.8,
        eps_growth_yoy=0.12,
        fetched_at=datetime(2026, 6, 21, 12, 0, 0),
    )


def _make_macro() -> PolishMacroSnapshot:
    return PolishMacroSnapshot(
        eur_pln=Decimal("4.30"),
        usd_pln=Decimal("3.95"),
        eur_pln_30d_change_pct=Decimal("0.021"),
        usd_pln_30d_change_pct=Decimal("0.048"),
        fetched_at=datetime(2026, 6, 21, 12, 0, 0),
    )


def _find_tool(tools: list[Tool], name: str) -> Tool:
    matches = [t for t in tools if t.name == name]
    assert len(matches) == 1, f"oczekiwano dokładnie jednego toola {name!r}"
    return matches[0]


def _assert_json_serializable(payload: dict[str, object]) -> None:
    """Każda wartość zmapowana na typ JSON-native (lub kontener takich)."""
    import json

    json.dumps(payload)  # rzuci TypeError gdyby został Decimal/dataclass


# --- Kształt fabryki -------------------------------------------------------


def test_build_research_tools_returns_expected_tool_names() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    macro_port = Mock(spec=MacroIndicatorsPort)

    tools = build_research_tools(fundamentals_port, macro_port)

    names = {t.name for t in tools}
    assert names == {"get_fundamentals", "get_macro"}
    assert all(isinstance(t, Tool) for t in tools)


def test_get_fundamentals_tool_has_well_formed_schema() -> None:
    tools = build_research_tools(
        Mock(spec=FundamentalsPort), Mock(spec=MacroIndicatorsPort)
    )
    tool = _find_tool(tools, "get_fundamentals")

    schema = tool.parameters
    assert schema["type"] == "object"
    assert "symbol" in schema["properties"]
    assert schema["properties"]["symbol"]["type"] == "string"
    assert schema["required"] == ["symbol"]
    assert isinstance(tool.description, str) and tool.description


def test_get_macro_tool_has_empty_object_schema() -> None:
    tools = build_research_tools(
        Mock(spec=FundamentalsPort), Mock(spec=MacroIndicatorsPort)
    )
    tool = _find_tool(tools, "get_macro")

    schema = tool.parameters
    assert schema["type"] == "object"
    # Brak wymaganych argumentów — pusty obiekt JSON Schema.
    assert schema.get("required", []) == []
    assert schema.get("properties", {}) == {}
    assert isinstance(tool.description, str) and tool.description


# --- get_fundamentals: happy path -----------------------------------------


def test_get_fundamentals_func_calls_port_and_maps_fields() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    fundamentals_port.get_fundamentals.return_value = _make_fundamentals()
    tools = build_research_tools(fundamentals_port, Mock(spec=MacroIndicatorsPort))
    tool = _find_tool(tools, "get_fundamentals")

    result = tool.func({"symbol": "AAPL"})

    fundamentals_port.get_fundamentals.assert_called_once_with("AAPL")
    assert result["available"] is True
    assert result["trailing_pe"] == pytest.approx(28.5)
    assert result["forward_pe"] == pytest.approx(24.0)
    assert result["peg_ratio"] == pytest.approx(1.8)
    assert result["eps_growth_yoy"] == pytest.approx(0.12)
    # Pola liczbowe są plain float, nie Decimal/inne.
    for key in ("trailing_pe", "forward_pe", "peg_ratio", "eps_growth_yoy"):
        assert isinstance(result[key], float), key
    _assert_json_serializable(result)


def test_get_fundamentals_func_preserves_none_fields() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    fundamentals_port.get_fundamentals.return_value = Fundamentals(
        trailing_pe=15.0,
        forward_pe=None,
        peg_ratio=None,
        eps_growth_yoy=None,
        fetched_at=datetime(2026, 6, 21, 12, 0, 0),
    )
    tools = build_research_tools(fundamentals_port, Mock(spec=MacroIndicatorsPort))
    tool = _find_tool(tools, "get_fundamentals")

    result = tool.func({"symbol": "SPY"})

    assert result["available"] is True
    assert result["trailing_pe"] == pytest.approx(15.0)
    assert result["forward_pe"] is None
    assert result["peg_ratio"] is None
    assert result["eps_growth_yoy"] is None
    _assert_json_serializable(result)


# --- get_fundamentals: brzegi (None / wyjątek) ----------------------------


def test_get_fundamentals_func_returns_unavailable_when_port_returns_none() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    fundamentals_port.get_fundamentals.return_value = None
    tools = build_research_tools(fundamentals_port, Mock(spec=MacroIndicatorsPort))
    tool = _find_tool(tools, "get_fundamentals")

    result = tool.func({"symbol": "BTC"})

    assert result["available"] is False
    _assert_json_serializable(result)


def test_get_fundamentals_func_swallows_port_exception() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    fundamentals_port.get_fundamentals.side_effect = RuntimeError("API down")
    tools = build_research_tools(fundamentals_port, Mock(spec=MacroIndicatorsPort))
    tool = _find_tool(tools, "get_fundamentals")

    # Nie wolno rzucić — pętla tool-use dostaje czysty dict.
    result = tool.func({"symbol": "AAPL"})

    assert result["available"] is False
    _assert_json_serializable(result)


def test_get_fundamentals_func_handles_missing_symbol_arg() -> None:
    fundamentals_port = Mock(spec=FundamentalsPort)
    tools = build_research_tools(fundamentals_port, Mock(spec=MacroIndicatorsPort))
    tool = _find_tool(tools, "get_fundamentals")

    # Brak "symbol" w args — defensywny func nie wybucha.
    result = tool.func({})

    assert result["available"] is False
    fundamentals_port.get_fundamentals.assert_not_called()
    _assert_json_serializable(result)


# --- get_macro: happy path + brzegi ---------------------------------------


def test_get_macro_func_calls_port_and_maps_fields() -> None:
    macro_port = Mock(spec=MacroIndicatorsPort)
    macro_port.fetch_polish_macro.return_value = _make_macro()
    tools = build_research_tools(Mock(spec=FundamentalsPort), macro_port)
    tool = _find_tool(tools, "get_macro")

    result = tool.func({})

    macro_port.fetch_polish_macro.assert_called_once_with()
    assert result["available"] is True
    assert result["eur_pln"] == pytest.approx(4.30)
    assert result["usd_pln"] == pytest.approx(3.95)
    assert result["eur_pln_30d_change_pct"] == pytest.approx(0.021)
    assert result["usd_pln_30d_change_pct"] == pytest.approx(0.048)
    # Żaden Decimal nie wyciekł.
    for key in (
        "eur_pln",
        "usd_pln",
        "eur_pln_30d_change_pct",
        "usd_pln_30d_change_pct",
    ):
        assert isinstance(result[key], float), key
        assert not isinstance(result[key], Decimal), key
    _assert_json_serializable(result)


def test_get_macro_func_returns_unavailable_when_port_returns_none() -> None:
    macro_port = Mock(spec=MacroIndicatorsPort)
    macro_port.fetch_polish_macro.return_value = None
    tools = build_research_tools(Mock(spec=FundamentalsPort), macro_port)
    tool = _find_tool(tools, "get_macro")

    result = tool.func({})

    assert result["available"] is False
    _assert_json_serializable(result)


def test_get_macro_func_swallows_port_exception() -> None:
    macro_port = Mock(spec=MacroIndicatorsPort)
    macro_port.fetch_polish_macro.side_effect = RuntimeError("NBP timeout")
    tools = build_research_tools(Mock(spec=FundamentalsPort), macro_port)
    tool = _find_tool(tools, "get_macro")

    result = tool.func({})

    assert result["available"] is False
    _assert_json_serializable(result)
