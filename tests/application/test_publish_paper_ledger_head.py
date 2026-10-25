"""WIA-59: głowa paper ledgera raz dziennie obok atestacji.

Use case zna tylko porty: repozytorium (głowa łańcucha) i publisher głowy
(append-only JSONL). Drugi przebieg tego samego dnia nic nie dopisuje."""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

from src.application.errors import ProviderUnavailable
from src.application.ports import LedgerHeadPublisherPort, RepositoryPort
from src.application.use_cases.publish_paper_ledger_head import PublishPaperLedgerHeadUseCase
from src.domain.paper_ledger import LedgerHead

TODAY = date(2026, 10, 19)


def _ports(*, last: date | None, head: LedgerHead | None) -> tuple[Mock, Mock]:
    repository = Mock(spec=RepositoryPort)
    repository.get_paper_ledger_head.return_value = head
    publisher = Mock(spec=LedgerHeadPublisherPort)
    publisher.last_published_on.return_value = last
    publisher.publish_head.return_value = "public/attestation/paper_ledger_heads.jsonl"
    return repository, publisher


def test_first_run_of_the_day_publishes_the_head() -> None:
    repository, publisher = _ports(
        last=date(2026, 10, 16), head=LedgerHead(seq=7, row_hash="a" * 64)
    )

    published = PublishPaperLedgerHeadUseCase(repository, publisher).run(TODAY)

    assert published == "public/attestation/paper_ledger_heads.jsonl"
    publisher.publish_head.assert_called_once_with(
        {
            "kind": "paper_ledger_head",
            "ledger_version": 1,
            "published_on": "2026-10-19",
            "seq": 7,
            "head": "a" * 64,
        }
    )


def test_second_run_of_the_day_publishes_nothing() -> None:
    repository, publisher = _ports(last=TODAY, head=LedgerHead(seq=7, row_hash="a" * 64))

    assert PublishPaperLedgerHeadUseCase(repository, publisher).run(TODAY) == ""
    publisher.publish_head.assert_not_called()
    repository.get_paper_ledger_head.assert_not_called()


def test_empty_ledger_publishes_nothing() -> None:
    repository, publisher = _ports(last=None, head=None)

    assert PublishPaperLedgerHeadUseCase(repository, publisher).run(TODAY) == ""
    publisher.publish_head.assert_not_called()


def test_unreadable_head_publishes_nothing() -> None:
    repository, publisher = _ports(last=None, head=None)
    repository.get_paper_ledger_head.side_effect = ProviderUnavailable("down")

    assert PublishPaperLedgerHeadUseCase(repository, publisher).run(TODAY) == ""
    publisher.publish_head.assert_not_called()
