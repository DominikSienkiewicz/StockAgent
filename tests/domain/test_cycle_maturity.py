"""Testy domeny dojrzałości cyklu — rozróżnienie "Dzień 1" od stanu ustalonego.

Kluczowy inwariant (największe ryzyko produktowe): cold-start (brak poprzedniej
ceny) NIE może być mylony z masową awarią źródeł. Cykl, w którym wszystkie
symbole padły z błędu, nie jest powitalnym "Dniem 1" — to incydent."""

from __future__ import annotations

from src.domain.cycle_maturity import CycleMaturity, SkipReason, classify_cycle


def test_skip_reason_members() -> None:
    # Kontrakt enuma przyczyn pominięcia (orkiestrator wystawi je na DTO).
    assert SkipReason.COLD_START.value == "COLD_START"
    assert SkipReason.BELOW_THRESHOLD.value == "BELOW_THRESHOLD"
    assert SkipReason.UNSUPPORTED_PRICE.value == "UNSUPPORTED_PRICE"


def test_cycle_maturity_members() -> None:
    assert CycleMaturity.FIRST_RUN.value == "FIRST_RUN"
    assert CycleMaturity.STEADY_STATE.value == "STEADY_STATE"


def test_all_cold_start_is_first_run() -> None:
    # Dzień 1: każdy obsługiwany symbol nie ma punktu odniesienia (cold-start).
    result = classify_cycle(
        saved_count=0,
        cold_start_count=45,
        below_threshold_count=0,
        error_count=0,
    )
    assert result is CycleMaturity.FIRST_RUN


def test_all_errors_is_not_first_run() -> None:
    # RYZYKO ZAMROŻONE: masowa awaria źródeł to NIE powitalny Dzień 1.
    result = classify_cycle(
        saved_count=0,
        cold_start_count=0,
        below_threshold_count=0,
        error_count=45,
    )
    assert result is not CycleMaturity.FIRST_RUN
    assert result is CycleMaturity.STEADY_STATE


def test_saved_prediction_means_history_exists() -> None:
    # Choćby jedna zapisana predykcja = istnieje już historia → nie Dzień 1.
    result = classify_cycle(
        saved_count=1,
        cold_start_count=44,
        below_threshold_count=0,
        error_count=0,
    )
    assert result is CycleMaturity.STEADY_STATE


def test_below_threshold_means_reference_price_existed() -> None:
    # "Poniżej progu" implikuje poprzednią cenę → punkt odniesienia istniał.
    result = classify_cycle(
        saved_count=0,
        cold_start_count=40,
        below_threshold_count=5,
        error_count=0,
    )
    assert result is CycleMaturity.STEADY_STATE


def test_cold_start_must_dominate_errors() -> None:
    # Garstka cold-startów tonie w masowej awarii → to incydent, nie Dzień 1.
    result = classify_cycle(
        saved_count=0,
        cold_start_count=2,
        below_threshold_count=0,
        error_count=40,
    )
    assert result is CycleMaturity.STEADY_STATE


def test_cold_start_survives_minor_errors() -> None:
    # Dzień 1 z paroma sieciowymi wpadkami wciąż jest Dniem 1 (cold-start > błędy).
    result = classify_cycle(
        saved_count=0,
        cold_start_count=43,
        below_threshold_count=0,
        error_count=2,
    )
    assert result is CycleMaturity.FIRST_RUN


def test_empty_cycle_is_steady_state() -> None:
    # Brak jakichkolwiek obsługiwanych symboli — degeneracja, nie powitanie.
    result = classify_cycle(
        saved_count=0,
        cold_start_count=0,
        below_threshold_count=0,
        error_count=0,
    )
    assert result is CycleMaturity.STEADY_STATE


# Wykluczenie SkipReason.UNSUPPORTED_PRICE z mianownika NIE jest już testowane
# tutaj: `classify_cycle` nie przyjmuje ich liczności, więc oba dawne testy
# ("cold-start + unsupported → FIRST_RUN", "sam unsupported → STEADY_STATE")
# po usunięciu martwego kwarga stały się dosłownymi duplikatami sąsiadów obok.
# Filtrowanie robi orkiestrator, więc kontrakt weryfikuje `TestCycleMaturityWiring`
# w tests/test_main_entrypoints.py — na realnych `SymbolResult`, nie na liczbach.
