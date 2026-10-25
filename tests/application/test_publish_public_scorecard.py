"""#10 — publikacja publicznego scorecardu jako use case (ARCH-02).

Use case zna tylko porty (repozytorium i publisher strony); konfigurację okna
i protokół ewaluacji dostaje w `PublicScorecardOptions` od composition root.
Wiring z `Settings` (flaga, druga instancja publishera) pilnuje
`tests/test_main_agent.py::TestPublicScorecard`.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.application.ports import RepositoryPort, WebPublisherPort
from src.application.report_evidence import PaperCosts
from src.application.use_cases.publish_public_scorecard import (
    PublicScorecardOptions,
    PublishPublicScorecardUseCase,
)
from src.domain.evaluation_protocol import EvaluationProtocol

_RESOLVED = [
    {
        "symbol": "NVDA",
        "predicted_trend": "BULLISH",
        "is_trend_correct": True,
        "price_at_prediction": "100",
        "actual_price_after_12h": "110",
        "timestamp": "2026-10-19T05:30:00+00:00",
    }
]


def _protocol() -> EvaluationProtocol:
    from src.config import load_evaluation_protocol

    return load_evaluation_protocol()


def _options() -> PublicScorecardOptions:
    return PublicScorecardOptions(
        window_days=30,
        settle_after_hours=12,
        symbols=["NVDA"],
        crypto_symbols=[],
        load_protocol=_protocol,
    )


def _repo(resolved: list[dict[str, object]] | None = None) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_resolved_predictions.return_value = _RESOLVED if resolved is None else resolved
    repo.get_resolved_for_calibration.return_value = [
        {"confidence_score": 0.8, "is_trend_correct": True, "predicted_trend": "BULLISH"},
    ]
    repo.get_price_history.return_value = []
    return repo


def test_publishes_the_rendered_page_of_the_window() -> None:
    repo = _repo()
    publisher = Mock(spec=WebPublisherPort)

    PublishPublicScorecardUseCase(repo, publisher, _options()).run()

    publisher.publish.assert_called_once()
    # Pokrycie danych (DAT-16) czyta rozliczenia drugi raz — okno jest to samo.
    assert {call.args for call in repo.get_resolved_predictions.call_args_list} == {(30,)}
    repo.get_resolved_for_calibration.assert_called_once_with(30)
    assert "NVDA" not in publisher.publish.call_args.args[0]


def test_empty_window_publishes_nothing() -> None:
    repo = _repo(resolved=[])
    repo.get_resolved_for_calibration.return_value = []
    publisher = Mock(spec=WebPublisherPort)

    PublishPublicScorecardUseCase(repo, publisher, _options()).run()

    publisher.publish.assert_not_called()


def test_read_error_is_swallowed_by_run_and_raised_by_render() -> None:
    repo = _repo()
    repo.get_resolved_predictions.side_effect = RuntimeError("supabase down")
    publisher = Mock(spec=WebPublisherPort)
    use_case = PublishPublicScorecardUseCase(repo, publisher, _options())

    use_case.run()

    publisher.publish.assert_not_called()
    with pytest.raises(RuntimeError, match="supabase down"):
        use_case.render()


def test_benchmark_is_read_only_with_costs() -> None:
    repo = _repo()
    use_case = PublishPublicScorecardUseCase(repo, Mock(spec=WebPublisherPort), _options())

    use_case.render()
    assert not _benchmark_reads(repo)

    use_case.render(PaperCosts(cost_bps=10.0))
    assert _benchmark_reads(repo) == [("IVV", 31)]


def _benchmark_reads(repo: Mock) -> list[tuple[object, ...]]:
    # Pokrycie danych (DAT-16) też czyta historię cen — liczymy tylko IVV.
    return [call.args for call in repo.get_price_history.call_args_list if call.args[0] == "IVV"]


def _json_publisher() -> Mock:
    publisher = Mock(spec=WebPublisherPort)
    publisher.publish.return_value = "public/scorecard/index.html"
    publisher.publish_asset.return_value = "public/scorecard/scorecard.json"
    return publisher


def test_publishes_scorecard_json_next_to_the_page() -> None:
    # WIA-81: ta sama publikacja daje stronę i jej maszynowy bliźniak.
    publisher = _json_publisher()

    PublishPublicScorecardUseCase(_repo(), publisher, _options()).run()

    name, content, content_type = publisher.publish_asset.call_args.args
    assert (name, content_type) == ("scorecard.json", "application/json")
    document = json.loads(content)
    assert document["n_resolved"] == 1
    assert "NVDA" not in content


def test_empty_window_publishes_no_json_either() -> None:
    repo = _repo(resolved=[])
    repo.get_resolved_for_calibration.return_value = []
    publisher = _json_publisher()

    assert PublishPublicScorecardUseCase(repo, publisher, _options()).publish() is None

    publisher.publish_asset.assert_not_called()


def test_publish_reports_a_failed_write() -> None:
    publisher = _json_publisher()
    publisher.publish_asset.return_value = ""

    assert PublishPublicScorecardUseCase(_repo(), publisher, _options()).publish() is False


def test_publish_reports_both_files_written() -> None:
    assert PublishPublicScorecardUseCase(_repo(), _json_publisher(), _options()).publish() is True
