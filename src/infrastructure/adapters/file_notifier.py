"""DX-04: kanał „mail” trybu offline — raport cyklu zapisany do plików.

Te same nazwy plików co podgląd `render-sample-report` (DX-01): treść HTML maila
w `report-email.html` i wersja tekstowa z linią tematu w `report-email.txt`.
Kolejny raport w tym samym katalogu nadpisuje poprzedni; `silent` nie ma tu
znaczenia (plik nie dzwoni)."""

from __future__ import annotations

from pathlib import Path

from src.application.ports import ReportNotifierPort

HTML_NAME = "report-email.html"
TEXT_NAME = "report-email.txt"


class FileReportNotifier(ReportNotifierPort):
    """`ReportNotifierPort` zapisujący raport do katalogu `out_dir`."""

    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    @property
    def html_path(self) -> Path:
        return self._out_dir / HTML_NAME

    @property
    def text_path(self) -> Path:
        return self._out_dir / TEXT_NAME

    def send_report(
        self, subject: str, html_body: str, plain_text: str, *, silent: bool = False
    ) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.html_path.write_text(html_body, encoding="utf-8")
        self.text_path.write_text(f"Temat: {subject}\n\n{plain_text}", encoding="utf-8")
