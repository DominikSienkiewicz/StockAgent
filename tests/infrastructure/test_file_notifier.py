"""DX-04: `FileReportNotifier` — mail cyklu offline zapisany do plików zamiast wysyłki."""

from __future__ import annotations

from pathlib import Path

from src.application.ports import ReportNotifierPort
from src.infrastructure.adapters.file_notifier import FileReportNotifier


def test_writes_the_html_body_and_the_text_with_its_subject(tmp_path: Path) -> None:
    notifier = FileReportNotifier(tmp_path / "cycle-1")

    notifier.send_report("StockAgent — NVDA +3,0%", "<html>raport</html>", "raport tekstowy")

    assert isinstance(notifier, ReportNotifierPort)
    assert notifier.html_path.read_text(encoding="utf-8") == "<html>raport</html>"
    assert notifier.text_path.read_text(encoding="utf-8") == (
        "Temat: StockAgent — NVDA +3,0%\n\nraport tekstowy"
    )
    assert notifier.html_path == tmp_path / "cycle-1" / "report-email.html"


def test_a_second_report_replaces_the_first(tmp_path: Path) -> None:
    notifier = FileReportNotifier(tmp_path)

    notifier.send_report("pierwszy", "<p>1</p>", "1")
    notifier.send_report("drugi", "<p>2</p>", "2", silent=True)

    assert notifier.html_path.read_text(encoding="utf-8") == "<p>2</p>"
