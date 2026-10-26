"""CLI „dlaczego?” — czytnik kwitów decyzyjnych predykcji (WIA-56).

`prediction_logs.decision_receipts` (migracja 020, R-13) był write-only: zapisywał
go graf i nikt go nie czytał, a audit trail, którego nikt nie czyta, nie jest
dowodem. To narzędzie składa z kwitu i głosów rady odpowiedź na pytanie, na czym
stała predykcja:

    - analogi RAG z id i podobieństwem (te, które naprawdę trafiły do promptu),
    - atrybucje cech SHAP-lite,
    - próg efektywny bramki po mnożnikach reżimu i earnings,
    - przepis predykcji (kwit v2) i ramiona eksperymentów,
    - głosy rady z `council_votes`.

Nieznana wersja schematu kwitu to komunikat, nie wyjątek — czytnik nie zgaduje.
Kwity istnieją tylko dla predykcji zapisanych z `receipts_enabled` (w praktyce
okno 2026-07-10…08-17), a format v1 nie niesie proweniencji.

Uruchom (ARCH-39: `Settings` i repozytorium składa `main_tools.py`):
    uv run python main_tools.py explain <prediction_id>
    uv run python main_tools.py explain --symbol NVDA --date 2026-08-14

Read-only, zero płatnych wywołań. Exit code: 0 zawsze (narzędzie raportujące).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date
from typing import Any

from src.application.decision_receipt import (
    SUPPORTED_RECEIPT_SCHEMAS,
    DecisionReceipt,
    effective_threshold_pct,
    read_decision_receipt,
    similarity_label,
)
from src.application.ports import RepositoryPort, StoredReceipt
from src.application.report_formatting import number, pct
from src.domain.council import InvestorOpinion

_INDENT = "  "


def explain_prediction(repository: RepositoryPort, prediction_id: str) -> str:
    """Kwit jednej predykcji po id albo komunikat, że takiej nie ma."""
    stored = repository.get_receipt(prediction_id)
    if stored is None:
        return f"❌ nie znaleziono predykcji {prediction_id}"
    return render_receipt(stored, repository.get_council_votes_for_prediction(prediction_id))


def explain_symbol_day(repository: RepositoryPort, symbol: str, day: date) -> str:
    """Kwity wszystkich predykcji symbolu z jednego dnia UTC, od najstarszej."""
    found = repository.find_receipts(symbol, day)
    if not found:
        return f"brak predykcji {symbol} z {day.isoformat()} (UTC)"
    return "\n\n".join(
        render_receipt(stored, repository.get_council_votes_for_prediction(stored.prediction_id))
        for stored in found
    )


def render_receipt(stored: StoredReceipt, votes: Sequence[InvestorOpinion]) -> str:
    """Nagłówek predykcji, treść kwitu (albo powód jej braku) i głosy rady."""
    lines = [_header(stored)]
    if stored.payload is None:
        lines.append(
            f"{_INDENT}brak kwitu decyzyjnego — predykcja zapisana bez receipts_enabled "
            "albo przed migracją 020"
        )
    else:
        receipt = read_decision_receipt(stored.payload)
        lines.extend(
            _unknown_schema_lines(stored.payload) if receipt is None else _receipt_lines(receipt)
        )
    lines.extend(_vote_lines(votes))
    return "\n".join(lines)


def _header(stored: StoredReceipt) -> str:
    moment = (
        "czas nieznany"
        if stored.predicted_at is None
        else stored.predicted_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )
    outcome = {True: "trafiona", False: "chybiona", None: "nierozliczona"}[stored.is_trend_correct]
    trend = stored.predicted_trend or "?"
    return (
        f"Predykcja {stored.prediction_id} · {stored.symbol} · {moment} · "
        f"prognoza {trend} · {outcome}"
    )


def _unknown_schema_lines(payload: Mapping[str, Any]) -> list[str]:
    known = ", ".join(str(version) for version in sorted(SUPPORTED_RECEIPT_SCHEMAS))
    return [
        f"{_INDENT}nieznana wersja schematu kwitu "
        f"(schema_version={payload.get('schema_version')!r}) — czytnik zna wersje {known}, "
        "treści nie zgaduje"
    ]


def _receipt_lines(receipt: DecisionReceipt) -> list[str]:
    threshold = effective_threshold_pct(receipt) or "brak w kwicie"
    lines = [
        f"{_INDENT}Kwit decyzyjny v{receipt.schema_version}",
        f"{_INDENT}Próg efektywny (po mnożnikach reżimu i earnings): {threshold}",
    ]
    lines.extend(_analog_lines(receipt.precedents))
    lines.extend(_attribution_lines(receipt.feature_attribution))
    if receipt.recipe:
        lines.append(f"{_INDENT}Przepis predykcji:")
        lines.extend(f"{_INDENT * 2}{key}: {value}" for key, value in receipt.recipe.items())
    if receipt.experiment_arms:
        arms = ", ".join(f"{name}={arm}" for name, arm in receipt.experiment_arms.items())
        lines.append(f"{_INDENT}Ramiona eksperymentów: {arms}")
    if receipt.model_used:
        lines.append(f"{_INDENT}Model, który odpowiedział: {receipt.model_used}")
    return lines


def _analog_lines(precedents: Sequence[Mapping[str, Any]]) -> list[str]:
    if not precedents:
        return [f"{_INDENT}Analogi RAG: brak"]
    lines = [f"{_INDENT}Analogi RAG ({len(precedents)}):"]
    for precedent in precedents:
        analog_id = precedent.get("id")
        parts = [f"#{analog_id}" if analog_id else "(bez id)"]
        similarity = similarity_label(precedent)
        if similarity is not None:
            parts.append(f"podobieństwo {similarity}")
        trend = precedent.get("predicted_trend")
        if trend:
            parts.append(str(trend))
        parts.append(_analog_outcome(precedent.get("is_trend_correct")))
        lines.append(_INDENT * 2 + " · ".join(parts))
    return lines


def _analog_outcome(is_trend_correct: object) -> str:
    if is_trend_correct is True:
        return "trafił"
    if is_trend_correct is False:
        return "chybił"
    return "wynik nieznany"


def _attribution_lines(attribution: Sequence[Mapping[str, Any]]) -> list[str]:
    if not attribution:
        return [f"{_INDENT}Atrybucje cech (SHAP-lite): brak"]
    lines = [f"{_INDENT}Atrybucje cech (SHAP-lite):"]
    for item in attribution:
        contribution = item.get("contribution")
        value = (
            number(contribution, decimals=4, signed=True)
            if isinstance(contribution, (int, float)) and not isinstance(contribution, bool)
            else "—"
        )
        lines.append(f"{_INDENT * 2}{item.get('feature', '?')}: {value}")
    return lines


def _vote_lines(votes: Sequence[InvestorOpinion]) -> list[str]:
    if not votes:
        return [f"{_INDENT}Głosy rady: brak zapisanych głosów"]
    lines = [f"{_INDENT}Głosy rady ({len(votes)}):"]
    for vote in votes:
        reasoning = f" — {vote.reasoning}" if vote.reasoning else ""
        lines.append(
            f"{_INDENT * 2}{vote.investor_name}: {vote.recommendation} "
            f"(pewność {pct(vote.confidence, decimals=0)}){reasoning}"
        )
    return lines
