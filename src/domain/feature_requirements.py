"""Domena: które migracje bazy wymagają które flagi (NIEZ-30).

Flaga zależna od migracji, włączona przed jej zaaplikowaniem, kończy się PGRST204
(zapis nieistniejącej kolumny) i gubi cały zapis predykcji. Mapa niżej mówi, od
której migracji flaga może być włączona; entry-point porównuje ją z wersją schematu,
którą zwraca `schema_version()` (migracja 071), i na czas przebiegu wyłącza flagi
bez swojej migracji.

Mapa to numer NAJNOWSZEJ migracji, której flaga potrzebuje (`attestation_enabled`
czeka na 043, nie na 024). Źródłem jest kolumna „Migration” tabeli flag w
`docs/configuration.md`; `tests/test_feature_requirements_wiring.py` pilnuje, że
mapa, tabela i pola `Settings` się nie rozjeżdżają. Flaga bez migracji nie ma tu wpisu.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

FLAG_REQUIRED_MIGRATION: Mapping[str, int] = MappingProxyType(
    {
        "subscriptions_enabled": 14,
        "persona_weighting_enabled": 15,
        "calibration_judge_enabled": 16,
        "regime_tagging_enabled": 17,
        "vector_memory_enabled": 17,
        "model_scorecard_enabled": 19,
        "receipts_enabled": 20,
        "shock_alerts_enabled": 21,
        "alpha_fusion_enabled": 22,
        "portfolio_positions_enabled": 23,
        "feature_set_filter_enabled": 27,
        "feature_set_version_enabled": 27,
        "model_scorecard_baselines_enabled": 28,
        "settled_at_windows_enabled": 29,
        "recipe_version_enabled": 30,
        "feature_snapshots_enabled": 31,
        "market_features_enabled": 31,
        "snapshot_training_enabled": 31,
        "interval_coverage_enabled": 32,
        "regime_hysteresis_enabled": 33,
        "gate_delta_enabled": 34,
        "fx_rates_cache_enabled": 36,
        "experiment_arms_enabled": 37,
        "cost_ledger_enabled": 40,
        "cycle_runs_enabled": 41,
        "reference_signal_enabled": 42,
        "attestation_enabled": 43,
        "calibration_score_version_enabled": 44,
        "persona_vote_outcomes_enabled": 45,
        "correction_category_enabled": 46,
        "model_scorecard_run_id_enabled": 49,
        "trend_probabilities_enabled": 51,
        "model_scorecard_feature_stats_enabled": 52,
        "earnings_cache_enabled": 53,
        "council_votes_upsert_enabled": 54,
        "stale_quote_enabled": 55,
        "fundamentals_market_cap_enabled": 56,
        "price_level_alerts_enabled": 57,
        "session_close_settlement_enabled": 59,
        "deribit_dvol_enabled": 25,
        "public_status_page_enabled": 41,
        "ml_settled_purge_enabled": 50,
        "model_scorecard_weights_sha256_enabled": 60,
        "digest_archive_enabled": 61,
        "split_detection_enabled": 62,
        "rag_exact_search_enabled": 63,
        "model_scorecard_provenance_enabled": 64,
        "persona_brier_ranking_enabled": 65,
        "executable_entry_enabled": 72,
        "paper_ledger_enabled": 73,
        "invalidation_level_enabled": 74,
        "shadow_predictions_enabled": 75,
        "rag_point_in_time_enabled": 76,
    }
)


def flags_missing_migration(enabled: Mapping[str, bool], *, schema_version: int) -> frozenset[str]:
    """Włączone flagi z `enabled` (nazwa → wartość), których migracja jest nowsza niż
    `schema_version`. Flaga wyłączona albo bez wpisu w mapie nigdy nie trafia do wyniku."""
    return frozenset(
        name
        for name, on in enabled.items()
        if on and FLAG_REQUIRED_MIGRATION.get(name, 0) > schema_version
    )
