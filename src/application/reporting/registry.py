"""ARCH-12: rejestr sekcji maila — kolejność, w jakiej mail je drukuje.

Jedna krotka dla HTML i wersji tekstowej: sekcja tylko w HTML ma `no_text`, więc
tekst idzie tą samą kolejnością. Nowa sekcja to wpis tutaj (patrz `section.py`).
"""

from __future__ import annotations

from src.application.report_accuracy_history import ACCURACY_HISTORY_SECTION
from src.application.report_alpha import ALPHA_SECTION
from src.application.report_backoffice import BACKOFFICE_SECTION
from src.application.report_consensus_shift import CONSENSUS_SHIFT_SECTION
from src.application.report_crypto import CRYPTO_SECTION
from src.application.report_header import (
    FIRST_SCREEN_SECTION,
    MASTHEAD_SECTION,
    SESSION_SECTION,
)
from src.application.report_lead import LEAD_SECTION, PREHEADER_SECTION
from src.application.report_onboarding import ONBOARDING_SECTION
from src.application.report_portfolio import PORTFOLIO_RADAR_SECTION
from src.application.report_predictions import PREDICTIONS_SECTION
from src.application.report_quota_banner import QUOTA_BANNER_SECTION
from src.application.report_resolved import RESOLVED_PREDICTIONS_SECTION
from src.application.report_risk_signals import RISK_SIGNALS_SECTION
from src.application.report_risk_watch import RISK_WATCH_SECTION
from src.application.report_skipped import (
    BUDGET_EXHAUSTED_SECTION,
    ERRORS_SECTION,
    IGNORED_SECTION,
    REBASELINED_SECTION,
)
from src.application.report_track_record import TRACK_RECORD_SECTION
from src.application.report_trade_signals import TRADE_SIGNALS_SECTION
from src.application.report_user_portfolio import USER_PORTFOLIO_SECTION
from src.application.reporting.section import ReportSection

REPORT_SECTIONS: tuple[ReportSection, ...] = (
    PREHEADER_SECTION,
    MASTHEAD_SECTION,
    QUOTA_BANNER_SECTION,
    LEAD_SECTION,
    ONBOARDING_SECTION,
    SESSION_SECTION,
    FIRST_SCREEN_SECTION,
    USER_PORTFOLIO_SECTION,
    RESOLVED_PREDICTIONS_SECTION,
    CONSENSUS_SHIFT_SECTION,
    REBASELINED_SECTION,
    TRADE_SIGNALS_SECTION,
    RISK_SIGNALS_SECTION,
    PREDICTIONS_SECTION,
    CRYPTO_SECTION,
    BACKOFFICE_SECTION,
    RISK_WATCH_SECTION,
    PORTFOLIO_RADAR_SECTION,
    ALPHA_SECTION,
    TRACK_RECORD_SECTION,
    ACCURACY_HISTORY_SECTION,
    BUDGET_EXHAUSTED_SECTION,
    IGNORED_SECTION,
    ERRORS_SECTION,
)
