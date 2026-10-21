"""DOC-17: diagram grafu w docs jest generowany z kodu i nie może kłamać.

Ręczny łańcuch w README rozjechał się z grafem (fundamenty warunkowe, węzeł
`ignore`). Strażnik porównuje zbiór węzłów i krawędzi skompilowanego grafu
z commitowanym plikiem `.mmd` i z blokiem mermaid w how-it-works — nie bajty,
bo format `draw_mermaid` zmienia się między wersjami LangGraph.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.tools.render_graph_diagram import (
    GRAPH_DIAGRAM_NAME,
    GRAPH_WITHOUT_FUNDAMENTALS_NAME,
    GraphTopology,
    graph_diagrams,
    parse_mermaid_topology,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


@pytest.mark.parametrize("name", [GRAPH_DIAGRAM_NAME, GRAPH_WITHOUT_FUNDAMENTALS_NAME])
def test_committed_diagram_matches_compiled_graph(name: str) -> None:
    committed = DOCS / name
    assert committed.is_file(), (
        f"docs/{name} nie istnieje — wygeneruj: uv run python main_tools.py render-graph-diagram"
    )

    expected = parse_mermaid_topology(graph_diagrams()[name])

    assert parse_mermaid_topology(committed.read_text(encoding="utf-8")) == expected


def test_how_it_works_embeds_the_production_diagram() -> None:
    text = (DOCS / "how-it-works.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.DOTALL)
    production = parse_mermaid_topology(graph_diagrams()[GRAPH_DIAGRAM_NAME])

    assert any(parse_mermaid_topology(block) == production for block in blocks)


def test_production_graph_reads_fundamentals_before_the_gate() -> None:
    topology = parse_mermaid_topology(graph_diagrams()[GRAPH_DIAGRAM_NAME])

    assert ("reflect", "fetch_fundamentals", False) in topology.edges
    assert {
        ("fetch_fundamentals", target, True)
        for target in ("analyze_sentiment", "ignore", "budget_exhausted")
    } <= topology.edges


def test_graph_without_fundamentals_gates_right_after_reflect() -> None:
    topology = parse_mermaid_topology(graph_diagrams()[GRAPH_WITHOUT_FUNDAMENTALS_NAME])

    assert "fetch_fundamentals" not in topology.nodes
    assert ("reflect", "ignore", True) in topology.edges


def test_parser_reads_nodes_and_both_edge_kinds() -> None:
    diagram = (
        "graph TD;\n"
        "\t__start__([<p>__start__</p>]):::first\n"
        "\tcheck_price(check_price)\n"
        "\t__start__ --> check_price;\n"
        "\tcheck_price -.-> ignore;\n"
        "\tcheck_price -. &nbsp;dalej&nbsp; .-> save;\n"
        "\tclassDef default fill:#f2f0ff,line-height:1.2\n"
    )

    assert parse_mermaid_topology(diagram) == GraphTopology(
        nodes=frozenset({"__start__", "check_price"}),
        edges=frozenset(
            {
                ("__start__", "check_price", False),
                ("check_price", "ignore", True),
                ("check_price", "save", True),
            }
        ),
    )
