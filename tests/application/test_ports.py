import inspect
import typing
from abc import ABC
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application import ports
from src.application.ports import (
    AdvisoryCouncilPort,
    FundamentalsPort,
    MLPredictionPort,
    RepositoryPort,
)
from src.domain.council import CouncilInput
from src.domain.model_scorecard import TrainingReport
from src.domain.value_objects import Percent

# TEST-20: każda klasa ABC zdefiniowana w module `ports` — nowy port wchodzi do testów
# abstrakcyjności bez dopisywania go ręcznie.
ALL_PORTS: list[type[ABC]] = sorted(
    (
        value
        for value in vars(ports).values()
        if isinstance(value, type) and issubclass(value, ABC) and value.__module__ == ports.__name__
    ),
    key=lambda port: port.__name__,
)


def test_all_ports_covers_every_abc_in_ports_module() -> None:
    # TEST-20: lista ręczna obejmowała 7 portów — nowy port wypadał z testu
    # abstrakcyjności. Filtr `__module__` odsiewa ABC zaimportowane z innych modułów,
    # a `issubclass(ABC)` — dataclassy i wyjątki zdefiniowane w `ports`.
    abc_classes: set[type] = {
        cls
        for _, cls in inspect.getmembers(ports, inspect.isclass)
        if issubclass(cls, ABC) and cls.__module__ == ports.__name__
    }

    assert set(ALL_PORTS) == abc_classes
    assert {MLPredictionPort, ports.CircuitStatePort, ports.ShadowModelPort} <= abc_classes
    assert ports.CycleRun not in abc_classes


@pytest.mark.parametrize("port_cls", ALL_PORTS)
def test_port_cannot_be_instantiated_directly(port_cls: type) -> None:
    with pytest.raises(TypeError, match="abstract"):
        port_cls()


@pytest.mark.parametrize("port_cls", ALL_PORTS)
def test_port_has_at_least_one_abstract_method(port_cls: type) -> None:
    assert len(port_cls.__abstractmethods__) >= 1  # type: ignore[attr-defined]


def test_advisory_council_port_is_abstract() -> None:
    assert inspect.isabstract(AdvisoryCouncilPort)


def test_advisory_council_port_can_be_mocked() -> None:
    mock = Mock(spec=AdvisoryCouncilPort)
    data = CouncilInput(
        symbol="AAPL",
        current_price=Decimal("180.00"),
        price_delta_pct=Percent(Decimal("3.5")),
        sentiment_score=0.6,
        news_articles=("headline",),
        llm_trend="BULLISH",
        llm_confidence=0.8,
        ml_price_target=Decimal("185.00"),
    )
    mock.analyze("AAPL", data)
    mock.analyze.assert_called_once()


def test_fundamentals_port_has_get_fundamentals() -> None:
    method = FundamentalsPort.get_fundamentals
    sig = inspect.signature(method)
    assert "symbol" in sig.parameters


def test_repository_port_has_fundamentals_methods() -> None:
    assert hasattr(RepositoryPort, "get_cached_fundamentals")
    assert hasattr(RepositoryPort, "save_fundamentals")


def test_repository_port_has_save_council_votes() -> None:
    # Strukturalny audit trail rady doradczej — osobna tabela na głosy
    # umożliwia odpytywanie "jak Burry głosował na NVDA w ostatnim miesiącu"
    # bez parsowania JSONB blob z prediction_logs.council_verdict.
    assert hasattr(RepositoryPort, "save_council_votes")
    sig = inspect.signature(RepositoryPort.save_council_votes)
    assert "prediction_id" in sig.parameters
    assert "votes" in sig.parameters


def test_ml_train_returns_domain_training_report() -> None:
    # ARCH-09: werdykt treningu przekracza port jako typ domenowy, nie jako
    # dict[str, Any] ze stringami, z którego nieznany status stawał się SKIPPED.
    hints = typing.get_type_hints(MLPredictionPort.train)

    assert hints["return"] is TrainingReport


def test_ml_train_takes_feature_rows_not_a_dataframe() -> None:
    # ARCH-09: application nie przekazuje DataFrame przez port — wiersze cech,
    # target i znaczniki czasu to zwykłe sekwencje; ramkę buduje adapter.
    hints = typing.get_type_hints(MLPredictionPort.train)

    assert hints["features"] == Sequence[Mapping[str, float | None]]
    assert hints["target"] == Sequence[float]
    assert hints["timestamps"] == Sequence[datetime] | None
