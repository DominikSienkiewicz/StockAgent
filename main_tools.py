"""Composition root narzędzi jednorazowych (ARCH-39).

Jedyne miejsce, gdzie narzędzie CLI dostaje `Settings` i konkretne adaptery.
Czysta logika (parsowanie, walidacja, diff, metryki, formatowanie) żyje
w `src/tools/` i nie składa adapterów — tutaj tylko łączymy zależności i wołamy
czyste funkcje. Każde narzędzie to jedna podkomenda argparse. Osobny
`main_<x>.py` dostaje wyłącznie entry-point z własnym workflow (Fast Loop,
Slow Loop, watch, deadman, backtest).

Użycie:
    uv run python main_tools.py --help
    uv run python main_tools.py <podkomenda> [opcje]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Parser z podkomendą na narzędzie; każda ustawia `handler` w defaults."""
    parser = argparse.ArgumentParser(
        prog="main_tools.py",
        description="StockAgent one-off tools (composition root, ARCH-39).",
    )
    parser.add_subparsers(dest="command", required=True, metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
