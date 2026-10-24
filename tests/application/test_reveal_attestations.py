"""#16 — sweep revealów commit-reveal jako use case (ARCH-02).

Przypadki przeniesione bez zmiany asercji z `tests/test_main_entrypoints.py`
(`TestAttestationRevealSweep`): sweep ujawnia WSZYSTKIE zapadłe commitmenty.
Bez niego predykcje symboli usuniętych z config.toml nigdy by się nie
odsłoniły, a dla sceptyka wygląda to jak ukrywanie nietrafień.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.application.ports import AttestationPublisherPort, RepositoryPort
from src.application.use_cases.reveal_attestations import RevealAttestationsUseCase


def _sweep(repo: MagicMock, pub: MagicMock) -> int:
    return RevealAttestationsUseCase(repo, pub).run()


class TestAttestationRevealSweep:
    @staticmethod
    def _row(pid: str, salt: str = "s" * 32) -> dict:
        return {
            "id": pid,
            "symbol": "GONE",
            "predicted_trend": "BULLISH",
            "predicted_target_price": "110.0",
            "price_at_prediction": "100.0",
            "actual_price_after_12h": "95.0",
            "is_trend_correct": False,
            "commitment_hash": "deadbeef",
            "commitment_salt": salt,
        }

    def test_reveals_every_matured_commitment(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1"), self._row("p-2")]
        pub = MagicMock()

        revealed = _sweep(repo, pub)

        assert revealed == 2
        assert pub.publish_reveal.call_count == 2
        assert repo.mark_commitment_revealed.call_count == 2

    def test_reveal_payload_exposes_the_salt(self) -> None:
        # Dopiero teraz sól wychodzi na jaw — to jest cały sens revealu.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock()

        _sweep(repo, pub)

        payload = pub.publish_reveal.call_args.args[0]
        assert payload["salt"] == "s" * 32
        assert payload["commitment"] == "deadbeef"

    def test_a_failed_reveal_does_not_stop_the_sweep(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1"), self._row("p-2")]
        pub = MagicMock()
        pub.publish_reveal.side_effect = [OSError("disk"), "path.jsonl"]

        assert _sweep(repo, pub) == 1

    def test_row_is_not_marked_revealed_when_publishing_fails(self) -> None:
        # Inaczej nietrafienie zniknęłoby na zawsze: oznaczone, nigdy ujawnione.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock()
        pub.publish_reveal.side_effect = OSError("disk")

        _sweep(repo, pub)

        repo.mark_commitment_revealed.assert_not_called()

    def test_row_is_not_marked_revealed_when_publisher_returns_empty(self) -> None:
        # Kontrakt portu: "" = wyłączone albo błąd. Adapter połyka OSError i zwraca
        # "", więc bez tej gałęzi wiersz dostałby `revealed_at` bez publikacji.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = ""

        assert _sweep(repo, pub) == 0
        repo.mark_commitment_revealed.assert_not_called()

    def test_reveal_is_not_counted_when_marking_fails(self) -> None:
        # Wynik sweepu to liczba wierszy z `revealed_at` — nieudany stempel
        # oznacza, że następny sweep ujawni ten wiersz jeszcze raz.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [self._row("p-1")]
        repo.mark_commitment_revealed.side_effect = Exception("db down")
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = "public/attestation/reveals.jsonl"

        assert _sweep(repo, pub) == 0

    def test_repository_error_degrades_to_no_sweep(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.side_effect = Exception("no column")

        assert _sweep(repo, MagicMock()) == 0

    @staticmethod
    def _sweep_one(row: dict) -> dict:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [row]
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = "public/attestation/reveals.jsonl"

        _sweep(repo, pub)

        return dict(pub.publish_reveal.call_args.args[0])

    def test_reveals_symbol_removed_from_config(self) -> None:
        # WIA-03: sweep nie zna watchlisty ani okna track recordu — commitment
        # symbolu wyrzuconego z config.toml też musi zostać ujawniony. Use case
        # nie dostaje Settings w ogóle, więc watchlista nie ma którędy wejść.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_unrevealed_commitments.return_value = [{**self._row("p-1"), "symbol": "VOO"}]
        pub = MagicMock(spec=AttestationPublisherPort)
        pub.publish_reveal.return_value = "public/attestation/reveals.jsonl"

        assert _sweep(repo, pub) == 1

        repo.get_unrevealed_commitments.assert_called_once_with()
        assert pub.publish_reveal.call_args.args[0]["symbol"] == "VOO"

    def test_reveal_publishes_stored_payload_verbatim(self) -> None:
        # WIA-70: weryfikacja bajt w bajt — niezależna od typów kolumn i od floatów,
        # w których PostgREST oddaje NUMERIC (100.0 zamiast "100.0000").
        import hashlib

        canonical = (
            '{"predicted_target_price":"110.0000","predicted_trend":"BULLISH",'
            '"price_at_prediction":"100.0000","schema_version":2,"symbol":"GONE"}'
        )
        salt = "s" * 32
        commitment = hashlib.sha256((canonical + salt).encode("utf-8")).hexdigest()
        row = {
            **self._row("p-1", salt),
            "commitment_hash": commitment,
            "commitment_payload": canonical,
            "predicted_target_price": 110.0,
            "price_at_prediction": 100.0,
        }

        published = self._sweep_one(row)

        assert published["payload"] == canonical
        recomputed = hashlib.sha256((published["payload"] + published["salt"]).encode("utf-8"))
        assert recomputed.hexdigest() == published["commitment"]

    def test_legacy_row_without_payload_is_revealed_from_columns(self) -> None:
        # Wiersze sprzed 043 (albo bez kolumny) ujawniamy z kolumn, jak dotąd.
        published = self._sweep_one({**self._row("p-1"), "commitment_payload": None})

        assert "payload" not in published
        assert published["predicted_target_price"] == "110.0"
        assert published["price_at_prediction"] == "100.0"
