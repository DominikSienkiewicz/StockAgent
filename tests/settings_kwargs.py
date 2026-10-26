"""Argument `_env_file=None` dla `Settings` w testach pod strict mypy.

Rozpakowanie tego słownika przekazuje `_env_file=None` wprost do runtime'u (test nie
czyta `.env` z dysku), a resztę argumentów konstruktora mypy dalej sprawdza. Przed
ARCH-22 mypy nie znał `_env_file` z `BaseSettings.__init__`; plugin `pydantic.mypy`
go zna, a słownik zostaje wspólną konwencją testów.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NO_ENV_FILE: Mapping[str, Any] = {"_env_file": None}
