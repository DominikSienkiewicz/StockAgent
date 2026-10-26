"""MAIL-35: polityka cichego dnia — pełny, zwarty albo żaden mail.

Cichy dzień = nic, co pełny mail miałby do powiedzenia ponad stan: pusty lead,
zero rozliczeń, zero szoków z watcha, żaden symbol za bramką (ani przeanalizowany,
ani odcięty budżetem), zero błędów i nie pierwszy cykl (onboarding). Tylko wtedy
polityka może skrócić albo pominąć mail.
"""

from __future__ import annotations

import pytest

from src.domain.digest_lead import LeadItem
from src.domain.digest_policy import DigestForm, DigestPolicy, should_send

LEAD = LeadItem(icon="⚡", headline="NVDA -4,2%", detail="rada PODZIELONA", symbol="NVDA")


def test_compact_policy_triggers_only_without_lead_and_resolved() -> None:
    policy = DigestPolicy.COMPACT_WHEN_QUIET

    assert should_send([], resolved=0, shocks=0, policy=policy) is DigestForm.COMPACT
    assert should_send([], resolved=1, shocks=0, policy=policy) is DigestForm.FULL
    assert should_send([LEAD], resolved=0, shocks=0, policy=policy) is DigestForm.FULL


def test_always_policy_sends_the_full_mail_even_on_a_quiet_day() -> None:
    assert should_send([], resolved=0, shocks=0, policy=DigestPolicy.ALWAYS) is DigestForm.FULL


def test_skip_policy_drops_only_the_quiet_day() -> None:
    policy = DigestPolicy.SKIP_WHEN_QUIET

    assert should_send([], resolved=0, shocks=0, policy=policy) is DigestForm.SKIP
    assert should_send([], resolved=2, shocks=0, policy=policy) is DigestForm.FULL


@pytest.mark.parametrize(
    "noise",
    [
        {"resolved": None},
        {"shocks": 1},
        {"gated": 1},
        {"errors": 1},
        {"onboarding": True},
    ],
    ids=["resolved-unknown", "shock", "gated-symbol", "error", "onboarding"],
)
def test_anything_beyond_the_state_keeps_the_full_mail(noise: dict[str, object]) -> None:
    resolved = noise.get("resolved", 0)
    shocks = noise.get("shocks", 0)
    gated = noise.get("gated", 0)
    errors = noise.get("errors", 0)
    onboarding = noise.get("onboarding", False)
    assert resolved is None or isinstance(resolved, int)
    assert isinstance(shocks, int) and isinstance(gated, int) and isinstance(errors, int)
    assert isinstance(onboarding, bool)

    for policy in (DigestPolicy.COMPACT_WHEN_QUIET, DigestPolicy.SKIP_WHEN_QUIET):
        form = should_send(
            [],
            resolved=resolved,
            shocks=shocks,
            policy=policy,
            gated=gated,
            errors=errors,
            onboarding=onboarding,
        )
        assert form is DigestForm.FULL


def test_policy_values_match_the_config_strings() -> None:
    assert [policy.value for policy in DigestPolicy] == [
        "always",
        "compact_when_quiet",
        "skip_when_quiet",
    ]
