"""WIA-59: append-only paper ledger z łańcuchem hashy.

Każdy wpis to pozycja papierowa otwarta przy zapisie predykcji; `row_hash` wiąże
kanoniczną treść wpisu z hashem poprzedniego (`prev_hash`), więc zmiana dowolnego
wcześniejszego wiersza rozjeżdża łańcuch od tego miejsca."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.domain.attestation import canonicalize
from src.domain.paper_ledger import (
    GENESIS_HASH,
    ExitReason,
    LedgerRow,
    PaperTradeEntry,
    TradeSide,
    chain_entry,
    close_trade,
    first_broken_row,
    row_hash,
    side_of,
    trend_of,
)

MONDAY = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)


def _entry(prediction_id: str = "pred-1", **overrides: object) -> PaperTradeEntry:
    fields: dict[str, object] = {
        "prediction_id": prediction_id,
        "symbol": "NVDA",
        "side": TradeSide.LONG,
        "entry_price": Decimal("100"),
        "entry_ts": MONDAY,
        "cost_bps": Decimal("10"),
    }
    fields.update(overrides)
    return PaperTradeEntry(**fields)  # type: ignore[arg-type]


def _chain(*entries: PaperTradeEntry) -> list[LedgerRow]:
    rows: list[LedgerRow] = []
    head = GENESIS_HASH
    for entry in entries:
        row = chain_entry(entry, head)
        rows.append(row)
        head = row.row_hash
    return rows


class TestRowHash:
    def test_row_hash_chains_previous_hash(self) -> None:
        first = chain_entry(_entry("pred-1"), GENESIS_HASH)
        second = chain_entry(_entry("pred-2"), first.row_hash)

        assert first.prev_hash == GENESIS_HASH
        assert second.prev_hash == first.row_hash
        expected = hashlib.sha256(
            (canonicalize(_entry("pred-2").payload()) + first.row_hash).encode("utf-8")
        ).hexdigest()
        assert second.row_hash == expected

    def test_same_entry_after_another_head_hashes_differently(self) -> None:
        assert row_hash(_entry(), GENESIS_HASH) != row_hash(_entry(), "f" * 64)

    def test_payload_uses_the_storage_scale_of_the_columns(self) -> None:
        # Hash liczony z wartości, które baza przechowa: cena DECIMAL(14,4), koszt
        # NUMERIC(8,2), waga NUMERIC(8,4), znacznik UTC z mikrosekundami.
        entry = _entry(
            entry_price=Decimal("100.123456"),
            entry_ts=datetime(2026, 10, 19, 7, 30, tzinfo=UTC).astimezone(),
        )

        assert entry.payload() == {
            "ledger_version": 1,
            "prediction_id": "pred-1",
            "symbol": "NVDA",
            "side": "long",
            "entry_price": "100.1235",
            "entry_ts": "2026-10-19T07:30:00.000000+00:00",
            "cost_bps": "10.00",
            "weight": "1.0000",
        }

    def test_naive_timestamp_is_utc(self) -> None:
        naive = _entry(entry_ts=datetime(2026, 10, 19, 5, 30))

        assert naive.payload()["entry_ts"] == _entry().payload()["entry_ts"]

    def test_golden_row_hash_is_frozen(self) -> None:
        # Kanonizacja wpisu to kontrakt danych: weryfikator za miesiąc odtwarza
        # ten sam hash. Zmiana tego testu = nowa `ledger_version`.
        assert row_hash(_entry(), GENESIS_HASH) == GOLDEN_FIRST_ROW_HASH


class TestChainVerification:
    def test_intact_chain_verifies(self) -> None:
        assert first_broken_row(_chain(_entry("a"), _entry("b"), _entry("c"))) is None

    def test_tampered_middle_row_breaks_verification(self) -> None:
        rows = _chain(_entry("a"), _entry("b"), _entry("c"))
        forged = dataclasses.replace(rows[1].entry, side=TradeSide.SHORT)
        rows[1] = dataclasses.replace(rows[1], entry=forged)

        assert first_broken_row(rows) == 1

    def test_rehashed_middle_row_breaks_the_next_link(self) -> None:
        # Fałszerz przelicza hash podmienionego wiersza — rozjeżdża się następny.
        rows = _chain(_entry("a"), _entry("b"), _entry("c"))
        forged = dataclasses.replace(rows[1].entry, entry_price=Decimal("90"))
        rows[1] = chain_entry(forged, rows[1].prev_hash)

        assert first_broken_row(rows) == 2

    def test_removed_row_breaks_the_chain(self) -> None:
        rows = _chain(_entry("a"), _entry("b"), _entry("c"))

        assert first_broken_row([rows[0], rows[2]]) == 1

    def test_window_verifies_from_its_own_first_link(self) -> None:
        rows = _chain(_entry("a"), _entry("b"), _entry("c"))

        assert first_broken_row(rows[1:], start=rows[0].row_hash) is None
        assert first_broken_row(rows[1:]) == 0


class TestSides:
    @pytest.mark.parametrize(
        ("trend", "side"),
        [("BULLISH", TradeSide.LONG), ("bearish", TradeSide.SHORT), ("SIDEWAYS", TradeSide.FLAT)],
    )
    def test_trend_maps_to_side_and_back(self, trend: str, side: TradeSide) -> None:
        assert side_of(trend) is side
        assert trend_of(side) == trend.upper()


class TestCloseTrade:
    AT = datetime(2026, 10, 20, 5, 30, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("side", "pnl"),
        [
            (TradeSide.LONG, Decimal("0.06")),
            (TradeSide.SHORT, Decimal("-0.06")),
            (TradeSide.FLAT, Decimal("0")),
        ],
    )
    def test_pnl_is_the_signed_gross_return(self, side: TradeSide, pnl: Decimal) -> None:
        close = close_trade(side, Decimal("100"), Decimal("106"), self.AT, ExitReason.SETTLED)

        assert close.pnl == pnl
        assert close.exit_price == Decimal("106.0000")
        assert close.exit_reason is ExitReason.SETTLED

    def test_entry_without_a_price_closes_flat(self) -> None:
        close = close_trade(
            TradeSide.LONG, Decimal("0"), Decimal("106"), self.AT, ExitReason.SETTLED
        )

        assert close.pnl == Decimal("0")


class TestLedgerCosts:
    def test_crypto_pays_its_own_cost(self) -> None:
        from src.domain.paper_ledger import LedgerCosts
        from src.domain.value_objects import AssetType

        costs = LedgerCosts(cost_bps=Decimal("10"), crypto_cost_bps=Decimal("25"))

        assert costs.for_asset(AssetType.CRYPTO) == Decimal("25")
        assert costs.for_asset(AssetType.STOCK) == Decimal("10")
        assert costs.for_asset(None) == Decimal("10")


class TestHeadPublication:
    def test_head_record_names_the_day_the_position_and_the_hash(self) -> None:
        from datetime import date

        from src.domain.paper_ledger import LedgerHead, head_record

        record = head_record(LedgerHead(seq=42, row_hash="a" * 64), date(2026, 10, 19))

        assert record == {
            "kind": "paper_ledger_head",
            "ledger_version": 1,
            "published_on": "2026-10-19",
            "seq": 42,
            "head": "a" * 64,
        }

    def test_head_is_published_once_a_day(self) -> None:
        from datetime import date

        from src.domain.paper_ledger import should_publish_head

        today = date(2026, 10, 19)

        assert should_publish_head(None, today) is True
        assert should_publish_head(date(2026, 10, 16), today) is True
        assert should_publish_head(today, today) is False


GOLDEN_FIRST_ROW_HASH = "85111c38202501c1669a18530888dea152b2890f8e11a638f07bc5e5852f90f3"
