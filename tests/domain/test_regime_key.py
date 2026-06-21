"""Testy domeny kanonicznego klucza reżimu (Cross-Asset Vector Memory).

`canonical_regime_key` normalizuje dowolną etykietę reżimu do zwięzłego,
porównywalnego klucza, a `regimes_match` decyduje, czy dwa reżimy są zgodne
po kanonizacji. Niezmienniki:

- pusty / whitespace / None → `UNKNOWN_REGIME`;
- normalizacja: trim + upper-case + spacje → podkreślenia, przycięcie do
  `_MAX_KEY_LEN` (32) znaków;
- idempotencja: `canonical(canonical(x)) == canonical(x)`;
- `regimes_match`: UNKNOWN/None pasuje do wszystkiego, reszta tylko do siebie.
"""

from __future__ import annotations

import pytest

from src.domain.regime_key import (
    UNKNOWN_REGIME,
    canonical_regime_key,
    regimes_match,
)


class TestCanonicalRegimeKey:
    def test_none_maps_to_unknown(self) -> None:
        assert canonical_regime_key(None) == UNKNOWN_REGIME

    def test_empty_string_maps_to_unknown(self) -> None:
        assert canonical_regime_key("") == UNKNOWN_REGIME

    @pytest.mark.parametrize("blank", [" ", "   ", "\t", "\n", " \t\n "])
    def test_whitespace_only_maps_to_unknown(self, blank: str) -> None:
        assert canonical_regime_key(blank) == UNKNOWN_REGIME

    def test_risk_off_label_is_normalized(self) -> None:
        # "Risk Off" → upper-case + spacja na podkreślenie.
        assert canonical_regime_key("Risk Off") == "RISK_OFF"

    def test_lowercase_is_upper_cased(self) -> None:
        assert canonical_regime_key("risk_off") == "RISK_OFF"

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        assert canonical_regime_key("  Risk Off  ") == "RISK_OFF"

    def test_multiple_spaces_each_become_underscore(self) -> None:
        assert canonical_regime_key("a b c") == "A_B_C"

    def test_long_label_is_truncated_to_32_chars(self) -> None:
        # Etykieta dłuższa niż limit kolumny zostaje przycięta do 32 znaków.
        key = canonical_regime_key("X" * 40)
        assert len(key) == 32
        assert key == "X" * 32

    def test_truncation_happens_after_normalization(self) -> None:
        # 40 spacji-rozdzielonych znaków → 40-znakowy upper, przycięty do 32.
        raw = " ".join("a" * 40)  # "a a a ... a" → "A_A_..._A"
        key = canonical_regime_key(raw)
        assert len(key) == 32

    def test_idempotent_on_plain_label(self) -> None:
        once = canonical_regime_key("Risk Off")
        assert canonical_regime_key(once) == once

    @pytest.mark.parametrize(
        "label",
        ["Risk Off", "risk on", "  Neutral  ", "X" * 40, "a b c d", "RISK_OFF"],
    )
    def test_idempotency_invariant(self, label: str) -> None:
        once = canonical_regime_key(label)
        assert canonical_regime_key(once) == once

    def test_already_canonical_is_unchanged(self) -> None:
        assert canonical_regime_key("RISK_OFF") == "RISK_OFF"


class TestRegimesMatch:
    def test_same_canonical_regimes_match(self) -> None:
        assert regimes_match("Risk Off", "risk_off") is True

    def test_different_regimes_do_not_match(self) -> None:
        assert regimes_match("Risk Off", "Risk On") is False

    def test_unknown_left_matches_anything(self) -> None:
        assert regimes_match(UNKNOWN_REGIME, "Risk Off") is True

    def test_unknown_right_matches_anything(self) -> None:
        assert regimes_match("Risk Off", UNKNOWN_REGIME) is True

    def test_none_left_matches_anything(self) -> None:
        assert regimes_match(None, "Risk Off") is True

    def test_none_right_matches_anything(self) -> None:
        assert regimes_match("Risk Off", None) is True

    def test_empty_string_matches_anything(self) -> None:
        # Pusty string kanonizuje się do UNKNOWN → pasuje do wszystkiego.
        assert regimes_match("", "Risk Off") is True

    def test_both_unknown_match(self) -> None:
        assert regimes_match(None, None) is True

    def test_same_explicit_regime_matches_itself(self) -> None:
        assert regimes_match("Risk On", "Risk On") is True
