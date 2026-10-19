"""Tożsamość przebiegu (MAIL-42): który run i który kod wygenerował wynik.

Budowana raz w DI (`main_*.py`) z env GitHub Actions i przekazywana dalej
jako wartość — stopka maila, a w WIA-20 przepis predykcji. Sama klasa nie
czyta środowiska: `from_environ` dostaje mapę od wołającego.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LOCAL = "local"
_SHORT_SHA_LENGTH = 7


@dataclass(frozen=True)
class RunContext:
    """Przebieg, z którego pochodzi raport lub predykcja.

    Pola:
        run_id: `GITHUB_RUN_ID` albo "local".
        git_sha: `GITHUB_SHA` albo "local".
        trigger: `GITHUB_EVENT_NAME` (schedule / workflow_dispatch) albo "local".
        run_url: link do logów przebiegu — tylko w GitHub Actions, inaczej None.
    """

    run_id: str = LOCAL
    git_sha: str = LOCAL
    trigger: str = LOCAL
    run_url: str | None = None

    @property
    def short_sha(self) -> str:
        return self.git_sha[:_SHORT_SHA_LENGTH]

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> RunContext:
        """Przebieg z env GitHub Actions; poza GHA (brak `GITHUB_ACTIONS=true`)
        zawsze kontekst lokalny — wyeksportowane ręcznie GITHUB_* nie udają runu."""
        if environ.get("GITHUB_ACTIONS") != "true":
            return cls()
        run_id = environ.get("GITHUB_RUN_ID") or LOCAL
        return cls(
            run_id=run_id,
            git_sha=environ.get("GITHUB_SHA") or LOCAL,
            trigger=environ.get("GITHUB_EVENT_NAME") or LOCAL,
            run_url=_actions_run_url(environ, run_id),
        )


def _actions_run_url(environ: Mapping[str, str], run_id: str) -> str | None:
    server = environ.get("GITHUB_SERVER_URL")
    repository = environ.get("GITHUB_REPOSITORY")
    if not server or not repository or run_id == LOCAL:
        return None
    return f"{server.rstrip('/')}/{repository}/actions/runs/{run_id}"
