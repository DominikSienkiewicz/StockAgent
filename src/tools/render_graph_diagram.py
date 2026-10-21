"""DOC-17: diagram grafu Fast Loopa generowany z kodu — czyste funkcje.

`graph_diagrams` składa `create_agent_graph` z atrapami portów (węzły nie są
wykonywane, LangGraph rysuje tylko topologię) w obu wariantach: produkcyjnym
z `fetch_fundamentals` i bez niego. Zapis plików robi podkomenda
`main_tools.py render-graph-diagram` (ARCH-39). Format `draw_mermaid` zmienia
się między wersjami LangGraph, więc strażnik porównuje węzły i krawędzie
(`parse_mermaid_topology`), nie bajty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from src.application.agent_graph import AgentGraphDeps, create_agent_graph
from src.domain.value_objects import Threshold

GRAPH_DIAGRAM_NAME = "graph.mmd"
GRAPH_WITHOUT_FUNDAMENTALS_NAME = "graph-without-fundamentals.mmd"

_NODE = re.compile(r"^\s*(\w+)\s*[(\[]")
_EDGE = re.compile(r"^\s*(\w+)\s+(-->|-\.->|--\s.*?\s-->|-\.\s.*?\s\.->)\s*(\w+)\s*;?\s*$")


@dataclass(frozen=True)
class GraphTopology:
    """Węzły i krawędzie diagramu; krawędź to (źródło, cel, czy warunkowa)."""

    nodes: frozenset[str]
    edges: frozenset[tuple[str, str, bool]]


def _placeholder() -> Any:
    """Atrapa portu: rysowanie grafu nie woła żadnej metody portu."""
    return cast(Any, object())


def _deps(*, with_fundamentals: bool) -> AgentGraphDeps:
    port = _placeholder()
    return AgentGraphDeps(
        market_port=port,
        sentiment_port=port,
        news_port=port,
        repository_port=port,
        ml_port=port,
        llm_port=port,
        threshold=Threshold(Decimal("0.02")),
        fundamentals_port=port if with_fundamentals else None,
    )


def graph_diagrams() -> dict[str, str]:
    """Nazwa pliku → diagram mermaid grafu Fast Loopa (z fundamentami i bez)."""
    return {
        name: create_agent_graph(_deps(with_fundamentals=with_fundamentals))
        .compile()
        .get_graph()
        .draw_mermaid()
        for name, with_fundamentals in (
            (GRAPH_DIAGRAM_NAME, True),
            (GRAPH_WITHOUT_FUNDAMENTALS_NAME, False),
        )
    }


def parse_mermaid_topology(diagram: str) -> GraphTopology:
    """Węzły i krawędzie z tekstu `graph TD` (krawędź kropkowana = warunkowa)."""
    nodes: set[str] = set()
    edges: set[tuple[str, str, bool]] = set()
    for line in diagram.splitlines():
        edge = _EDGE.match(line)
        if edge:
            source, arrow, target = edge.groups()
            edges.add((source, target, arrow.startswith("-.")))
            continue
        node = _NODE.match(line)
        if node and node.group(1) not in {"graph", "classDef"}:
            nodes.add(node.group(1))
    return GraphTopology(nodes=frozenset(nodes), edges=frozenset(edges))
