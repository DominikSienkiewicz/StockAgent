"""Testy CompositeNotifier — fan-out send_report do wielu kanałów."""

from __future__ import annotations

from unittest.mock import Mock

from src.application.ports import ReportNotifierPort
from src.infrastructure.adapters.composite_notifier import CompositeNotifier


class TestCompositeNotifier:
    def test_implements_port(self) -> None:
        assert issubclass(CompositeNotifier, ReportNotifierPort)

    def test_fans_out_to_all_children(self) -> None:
        a = Mock(spec=ReportNotifierPort)
        b = Mock(spec=ReportNotifierPort)
        composite = CompositeNotifier([a, b])

        composite.send_report("Subj", "<p>H</p>", "plain")

        a.send_report.assert_called_once_with("Subj", "<p>H</p>", "plain")
        b.send_report.assert_called_once_with("Subj", "<p>H</p>", "plain")

    def test_one_failing_child_does_not_stop_others(self) -> None:
        good_before = Mock(spec=ReportNotifierPort)
        bad = Mock(spec=ReportNotifierPort)
        bad.send_report.side_effect = RuntimeError("channel down")
        good_after = Mock(spec=ReportNotifierPort)
        composite = CompositeNotifier([good_before, bad, good_after])

        # Wyjątek jednego kanału nie może przerwać pozostałych ani propagować.
        composite.send_report("S", "H", "T")

        good_before.send_report.assert_called_once()
        bad.send_report.assert_called_once()
        good_after.send_report.assert_called_once()

    def test_empty_list_is_noop(self) -> None:
        composite = CompositeNotifier([])
        composite.send_report("S", "H", "T")  # nie rzuca
