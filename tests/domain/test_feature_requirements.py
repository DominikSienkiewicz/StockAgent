"""NIEZ-30: flaga zależna od migracji nie może zostać włączona na bazie, która jej nie ma.

Flaga włączona przed migracją kończy się PGRST204 i gubi cały zapis predykcji.
Domena zna tylko mapę „flaga → numer migracji” i regułę wyłączania; skąd pochodzi
wersja schematu, wie wyłącznie entry-point.
"""

from __future__ import annotations

from src.domain.feature_requirements import (
    FLAG_REQUIRED_MIGRATION,
    flags_missing_migration,
)


def test_a_flag_whose_migration_is_missing_is_switched_off() -> None:
    assert flags_missing_migration({"receipts_enabled": True}, schema_version=19) == frozenset(
        {"receipts_enabled"}
    )


def test_a_flag_whose_migration_is_applied_stays_on() -> None:
    assert flags_missing_migration({"receipts_enabled": True}, schema_version=20) == frozenset()
    assert flags_missing_migration({"receipts_enabled": True}, schema_version=71) == frozenset()


def test_a_flag_that_is_off_is_never_reported() -> None:
    assert flags_missing_migration({"receipts_enabled": False}, schema_version=0) == frozenset()


def test_a_flag_without_a_migration_is_never_reported() -> None:
    assert flags_missing_migration({"model_routing_enabled": True}, schema_version=0) == frozenset()


def test_only_the_flags_beyond_the_schema_are_reported() -> None:
    enabled = {
        "receipts_enabled": True,  # 020
        "shock_alerts_enabled": True,  # 021
        "model_scorecard_enabled": True,  # 019
        "model_routing_enabled": True,
    }

    assert flags_missing_migration(enabled, schema_version=20) == frozenset(
        {"shock_alerts_enabled"}
    )


def test_the_map_carries_the_flags_the_decision_named() -> None:
    expected = {
        "receipts_enabled": 20,
        "vector_memory_enabled": 17,
        "shock_alerts_enabled": 21,
        "portfolio_positions_enabled": 23,
        "model_scorecard_enabled": 19,
        "alpha_fusion_enabled": 22,
    }

    assert {name: FLAG_REQUIRED_MIGRATION[name] for name in expected} == expected


def test_a_flag_that_needs_two_migrations_waits_for_the_later_one() -> None:
    # attestation_enabled: 024 (ledger) i 043 (payload commitmentu).
    assert FLAG_REQUIRED_MIGRATION["attestation_enabled"] == 43
    assert flags_missing_migration({"attestation_enabled": True}, schema_version=42) == frozenset(
        {"attestation_enabled"}
    )
