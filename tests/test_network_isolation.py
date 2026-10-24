"""TEST-11: testy jednostkowe nie mają dostępu do sieci (pytest-socket).

Nic poza dyscypliną mocków nie gwarantowało, że test jednostkowy nie wyjdzie
do sieci — a z wyeksportowanymi kluczami taki test płaci za API. `addopts`
w `pyproject.toml` wyłącza gniazda internetowe w każdym teście; testy
oznaczone `integration` albo `containers` dostają sieć z powrotem
(`tests/conftest.py`), a gniazda unixowe zostają dla klienta Dockera.
"""

from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError

from tests import conftest

# RFC 5737 (TEST-NET-1): adres dokumentacyjny — bez DNS, bez realnego celu.
_DOCUMENTATION_ADDRESS = ("192.0.2.1", 80)


class _CollectedItem:
    """Minimalny zastępnik `pytest.Item` dla hooka kolekcji."""

    def __init__(self, *markers: str) -> None:
        self._markers = frozenset(markers)
        self.added: list[str] = []

    def get_closest_marker(self, name: str) -> str | None:
        return name if name in self._markers else None

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker.name)


# pytest-socket ostrzega przy każdej blokadzie — tu blokada jest oczekiwana.
@pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")
def test_unit_tests_cannot_open_sockets() -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(_DOCUMENTATION_ADDRESS, timeout=1)


def test_unix_sockets_stay_available_for_the_docker_client() -> None:
    left, right = socket.socketpair()
    left.close()
    right.close()


@pytest.mark.parametrize("marker", ["integration", "containers"])
def test_network_markers_get_their_sockets_back(marker: str) -> None:
    item = _CollectedItem(marker)

    conftest.pytest_collection_modifyitems([item])  # type: ignore[list-item]

    assert item.added == ["enable_socket"]


def test_unmarked_tests_stay_offline() -> None:
    item = _CollectedItem("parametrize")

    conftest.pytest_collection_modifyitems([item])  # type: ignore[list-item]

    assert item.added == []
