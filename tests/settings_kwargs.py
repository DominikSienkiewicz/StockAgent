"""Argument `_env_file=None` dla `Settings` w testach pod strict mypy.

mypy syntezuje `__init__` modelu z jego pól i nie zna `_env_file` z
`BaseSettings.__init__`, więc `Settings(_env_file=None, ...)` to błąd `call-arg`.
Rozpakowanie tego słownika przekazuje `_env_file=None` wprost do runtime'u (test nie
czyta `.env` z dysku), a resztę argumentów konstruktora mypy dalej sprawdza.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

NO_ENV_FILE: Mapping[str, Any] = {"_env_file": None}
