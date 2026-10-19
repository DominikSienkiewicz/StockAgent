"""`main_tools.py` — composition root narzędzi jednorazowych (ARCH-39).

Narzędzia CLI dostają `Settings` i adaptery wyłącznie tutaj; `src/tools/` trzyma
czystą logikę. Każde narzędzie to podkomenda argparse.
"""

from __future__ import annotations

import pytest

import main_tools


def test_main_tools_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_tools.main([])

    assert exit_info.value.code == 2
    assert "<command>" in capsys.readouterr().err


def test_main_tools_help_names_the_composition_root(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_tools.main(["--help"])

    assert exit_info.value.code == 0
    assert "main_tools.py" in capsys.readouterr().out
