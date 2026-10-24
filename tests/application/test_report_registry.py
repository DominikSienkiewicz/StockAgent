"""ARCH-12: rejestr sekcji maila zamiast slotów w `report_builder`.

Mail to krotka `REPORT_SECTIONS` w kolejności, w jakiej go drukuje — jedna dla
HTML i wersji tekstowej. Każda sekcja ma renderer HTML i tekstu (sekcja tylko
w HTML świadomie bierze `no_text`). Że złożenie z rejestru daje mail bajt w bajt
jak dawne sloty, pilnuje golden snapshot (`test_report_builder_snapshot.py`).
"""

from __future__ import annotations

from src.application.reporting.registry import REPORT_SECTIONS

MAIL_ORDER = (
    "preheader",
    "masthead",
    "quota_banner",
    "lead",
    "onboarding",
    "session",
    "first_screen",
    "user_portfolio",
    "resolved_predictions",
    "consensus_shift",
    "rebaselined",
    "trade_signals",
    "risk_signals",
    "predictions",
    "crypto",
    "backoffice",
    "risk_watch",
    "portfolio_radar",
    "alpha",
    "track_record",
    "accuracy_history",
    "model_scorecard",
    "persona_leaderboard",
    "council_history",
    "mood",
    "delta_chart",
    "reflections",
    "suggestions",
    "correlation_chart",
    "finops",
    "budget_exhausted",
    "ignored",
    "errors",
)


def test_registry_covers_every_slot_with_html_and_text() -> None:
    assert tuple(section.key for section in REPORT_SECTIONS) == MAIL_ORDER
    for section in REPORT_SECTIONS:
        assert callable(section.render_html), section.key
        assert callable(section.render_text), section.key
