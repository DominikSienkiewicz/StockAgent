"""Testy domeny commit-reveal (`src/domain/attestation.py`).

Sedno tych testów to GOLDEN TESTY: zahardkodowane pary payload -> hash. Jeśli
ktokolwiek kiedykolwiek zmieni kanonizację (kolejność kluczy, separatory,
sposób serializacji `Decimal`, obecność `schema_version`), te literały zaczną
się rozjeżdżać z wynikiem i test zafailuje. To jedyna ochrona commitmentów
publikowanych DZIŚ, a weryfikowanych za wiele miesięcy — złamana wstecz
kanonizacja jest nie do naprawienia, więc musi być zamrożona od dnia 1.

Hashe poniżej policzono niezależnie od modułu (osobny skrypt mirrorujący
dokładnie kontrakt ze specyfikacji), żeby test był prawdziwym oraclem, a nie
tautologią „moduł zgadza się sam ze sobą".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, cast

import pytest

from src.domain import attestation

# --- Golden fixtures: payload -> oczekiwany hash (literały, NIE liczone w teście) ---

_SALT_A = "a" * 32
_SALT_B = "b" * 32
_SALT_C = "cafebabecafebabecafebabecafebabe"
_SALT_D = "d" * 32

_PAYLOAD_1 = {"symbol": "AAPL", "direction": "up", "target_price": Decimal("1.10")}
_CANON_1 = '{"direction":"up","schema_version":1,"symbol":"AAPL","target_price":"1.10"}'
_HASH_1 = "4dbc725a88e125189140003cd05d8a78c3efbc7522da93817df22476b01301b4"
_HASH_1_SALT_B = "cd3187a9bcf5babdd82c67b1369804816a8083a22d83ba7f22a801794426a856"

# Ten sam payload, inna kolejność kluczy — musi dać identyczny commitment.
_PAYLOAD_1_REORDERED = {"target_price": Decimal("1.10"), "symbol": "AAPL", "direction": "up"}

_PAYLOAD_3 = {
    "symbol": "MSFT",
    "horizon_hours": 12,
    "confidence": Decimal("0.75"),
    "rationale": "Zażółć gęślą jaźń",  # non-ASCII: ensure_ascii=False musi zostać
    "tags": ["ai", "momentum"],
    "meta": {"regime": "bull", "note": None},
}
_CANON_3 = (
    '{"confidence":"0.75","horizon_hours":12,"meta":{"note":null,"regime":"bull"},'
    '"rationale":"Zażółć gęślą jaźń","schema_version":1,"symbol":"MSFT",'
    '"tags":["ai","momentum"]}'
)
_HASH_3 = "8b21002b446abab04a8db003c980eccf0451ca0f9fead935a8a47c48ad6e3cf7"

# Round-trip Decimal: "1.10" != "1.1". Wybór projektowy — serializujemy przez
# str(Decimal), które ZACHOWUJE liczbę miejsc po przecinku. Dwie różne wartości
# Decimal muszą dać dwa różne commitmenty.
_PAYLOAD_TRAILING = {"price": Decimal("1.10")}
_HASH_TRAILING = "8b200357d9bc921dfa15a7230c78c972732d33ccf0f6e39ea623a39a5051b18a"
_PAYLOAD_NOTRAILING = {"price": Decimal("1.1")}
_HASH_NOTRAILING = "8fe2d8e40f9c6042c1018dfc1eed473b6354353dcd4560d47e32db46136bf92f"


# --- Golden: kanonizacja ---


def test_canonicalize_golden_flat() -> None:
    assert attestation.canonicalize(_PAYLOAD_1) == _CANON_1


def test_canonicalize_golden_nested_and_unicode() -> None:
    assert attestation.canonicalize(_PAYLOAD_3) == _CANON_3


def test_canonicalize_injects_schema_version() -> None:
    # schema_version MUSI trafić do kanonicznej reprezentacji nawet gdy payload go nie ma.
    assert '"schema_version":1' in attestation.canonicalize(_PAYLOAD_1)


def test_canonicalize_is_key_order_independent() -> None:
    assert attestation.canonicalize(_PAYLOAD_1) == attestation.canonicalize(_PAYLOAD_1_REORDERED)


def test_canonicalize_rejects_reserved_schema_version_key() -> None:
    # Payload nie może przemycić własnego schema_version — moduł jest właścicielem wersji.
    with pytest.raises(ValueError, match="schema_version"):
        attestation.canonicalize({"schema_version": 99, "symbol": "AAPL"})


def test_canonicalize_rejects_non_serializable_type() -> None:
    with pytest.raises(TypeError):
        attestation.canonicalize({"when": object()})


# --- Golden: commit ---


def test_commit_golden_flat() -> None:
    assert attestation.commit(_PAYLOAD_1, _SALT_A) == _HASH_1


def test_commit_golden_nested() -> None:
    assert attestation.commit(_PAYLOAD_3, _SALT_C) == _HASH_3


def test_commit_is_key_order_independent() -> None:
    ordered = attestation.commit(_PAYLOAD_1, _SALT_A)
    reordered = attestation.commit(_PAYLOAD_1_REORDERED, _SALT_A)
    assert ordered == reordered


def test_commit_changes_with_salt() -> None:
    assert attestation.commit(_PAYLOAD_1, _SALT_A) != attestation.commit(_PAYLOAD_1, _SALT_B)
    assert attestation.commit(_PAYLOAD_1, _SALT_B) == _HASH_1_SALT_B


def test_commit_is_sha256_hex() -> None:
    digest = attestation.commit(_PAYLOAD_1, _SALT_A)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# --- Round-trip precyzji Decimal ---


def test_decimal_trailing_zero_is_significant() -> None:
    # "1.10" i "1.1" to dwie różne reprezentacje -> dwa różne commitmenty.
    assert _HASH_TRAILING != _HASH_NOTRAILING
    assert attestation.commit(_PAYLOAD_TRAILING, _SALT_D) == _HASH_TRAILING
    assert attestation.commit(_PAYLOAD_NOTRAILING, _SALT_D) == _HASH_NOTRAILING


def test_decimal_serialized_as_string_not_float() -> None:
    # Gdyby Decimal poszedł przez float, precyzja by uciekła — kanoniczny string
    # musi zawierać dokładny literał "1.10", nie 1.1.
    assert '"1.10"' in attestation.canonicalize(_PAYLOAD_TRAILING)


# --- verify ---


def test_verify_accepts_matching_commitment() -> None:
    commitment = attestation.commit(_PAYLOAD_1, _SALT_A)
    assert attestation.verify(_PAYLOAD_1, _SALT_A, commitment) is True


def test_verify_accepts_reordered_payload() -> None:
    commitment = attestation.commit(_PAYLOAD_1, _SALT_A)
    assert attestation.verify(_PAYLOAD_1_REORDERED, _SALT_A, commitment) is True


def test_verify_rejects_tampered_payload() -> None:
    commitment = attestation.commit(_PAYLOAD_1, _SALT_A)
    tampered = dict(_PAYLOAD_1)
    tampered["direction"] = "down"
    assert attestation.verify(tampered, _SALT_A, commitment) is False


def test_verify_rejects_wrong_salt() -> None:
    commitment = attestation.commit(_PAYLOAD_1, _SALT_A)
    assert attestation.verify(_PAYLOAD_1, _SALT_B, commitment) is False


def test_verify_is_case_insensitive_on_hex() -> None:
    # Commitment z JSONL/GitHuba może przyjść wielkimi literami — nie odrzucaj go z tego powodu.
    commitment = attestation.commit(_PAYLOAD_1, _SALT_A)
    assert attestation.verify(_PAYLOAD_1, _SALT_A, commitment.upper()) is True


# --- new_salt ---


def test_new_salt_is_hex_and_unique() -> None:
    s1 = attestation.new_salt()
    s2 = attestation.new_salt()
    assert s1 != s2
    assert len(s1) >= 32
    assert all(c in "0123456789abcdef" for c in s1)


# --- Schemat v2 (WIA-02): ceny w skali DECIMAL(14,4) ---


def test_current_schema_is_v2_and_v1_stays_supported() -> None:
    # Jedno podbicie kontraktu; v1 zostaje do weryfikacji starych commitmentów.
    assert (attestation.SCHEMA_VERSION_V1, attestation.SCHEMA_VERSION) == (1, 2)


def test_canonicalize_injects_the_requested_schema_version() -> None:
    canonical = attestation.canonicalize(_PAYLOAD_1, schema_version=attestation.SCHEMA_VERSION)
    assert '"schema_version":2' in canonical


def test_v1_and_v2_commitments_of_the_same_payload_differ() -> None:
    v1 = attestation.commit(_PAYLOAD_1, _SALT_A)
    v2 = attestation.commit(_PAYLOAD_1, _SALT_A, schema_version=attestation.SCHEMA_VERSION)
    assert v1 == _HASH_1
    assert v2 != v1
    assert attestation.verify(
        _PAYLOAD_1, _SALT_A, v2, schema_version=attestation.SCHEMA_VERSION
    ) is True
    assert attestation.verify(_PAYLOAD_1, _SALT_A, v2) is False


def test_canonicalize_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        attestation.canonicalize(_PAYLOAD_1, schema_version=3)


def test_prediction_payload_quantizes_prices_like_decimal_14_4() -> None:
    # Python trzyma 90.11111111101, kolumna DECIMAL(14,4) zapisze 90.1111 —
    # hash musi powstać z wartości, którą baza przechowa (połówki od zera).
    payload = attestation.prediction_commitment_payload(
        symbol="AAPL",
        predicted_trend="BULLISH",
        predicted_target_price=Decimal("90.11111111101"),
        price_at_prediction=Decimal("90.11115"),
    )
    assert payload["predicted_target_price"] == "90.1111"
    assert payload["price_at_prediction"] == "90.1112"


def test_prediction_payload_uses_the_column_text_with_four_places() -> None:
    # Tekst kolumny NUMERIC(14,4) ma zawsze cztery miejsca: „100.0” → „100.0000”.
    payload = attestation.prediction_commitment_payload(
        symbol="AAPL",
        predicted_trend="SIDEWAYS",
        predicted_target_price=Decimal("100.0"),
        price_at_prediction=Decimal("100"),
    )
    assert payload["predicted_target_price"] == "100.0000"
    assert payload["price_at_prediction"] == "100.0000"


def test_prediction_payload_keeps_a_missing_target_as_null() -> None:
    payload = attestation.prediction_commitment_payload(
        symbol="AAPL",
        predicted_trend=None,
        predicted_target_price=None,
        price_at_prediction=Decimal("100"),
    )
    assert payload["predicted_target_price"] is None
    assert payload["predicted_trend"] is None


# --- Golden v2 (WIA-41): pewność, horyzont, werdykt rady, przepis, protokół ---
# Hashe policzone osobnym skryptem mirrorującym kontrakt v2 (json.dumps
# sort_keys + separatory + schema_version 2), NIE przez moduł.

_SALT_E = "e" * 32
_CANON_V2 = (
    '{"confidence_score":"0.8500","council":{"consensus_strength":"0.7143",'
    '"final_recommendation":"BUY"},"horizon_hours":12,"predicted_target_price":"131.2346",'
    '"predicted_trend":"BULLISH","price_at_prediction":"128.5000",'
    '"protocol_sha256":"' + "b" * 64 + '","recipe_version":"' + "a" * 64 + '",'
    '"schema_version":2,"symbol":"NVDA"}'
)
_HASH_V2 = "62f0e3d87b5c91662fc688fc59dd3a07e0acc7e7c2779f35e0d37acbec16004c"
_CANON_V2_BARE = (
    '{"confidence_score":null,"council":null,"horizon_hours":12,'
    '"predicted_target_price":null,"predicted_trend":"BULLISH",'
    '"price_at_prediction":"128.5000","protocol_sha256":null,"recipe_version":null,'
    '"schema_version":2,"symbol":"NVDA"}'
)
_HASH_V2_BARE = "5f97c292e94b4052cddf48a400f2dfda4c1047555649a3996fa00342c331f06d"


def _payload_v2(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "symbol": "NVDA",
        "predicted_trend": "BULLISH",
        "predicted_target_price": Decimal("131.23455"),
        "price_at_prediction": Decimal("128.5"),
        "confidence_score": 0.85,
        "horizon_hours": 12,
        "council_recommendation": "BUY",
        "council_consensus_strength": 0.714285,
        "recipe_version": "a" * 64,
        "protocol_sha256": "b" * 64,
    }
    fields.update(overrides)
    # Pola jak w sygnaturze funkcji; mypy strict (TEST-13) nie zawęża `**dict[str, object]`.
    return attestation.prediction_commitment_payload(**cast(dict[str, Any], fields))


def test_canonicalize_golden_v2() -> None:
    canonical = attestation.canonicalize(_payload_v2(), schema_version=attestation.SCHEMA_VERSION)
    assert canonical == _CANON_V2


def test_golden_hash_v2() -> None:
    commitment = attestation.commit(
        _payload_v2(), _SALT_E, schema_version=attestation.SCHEMA_VERSION
    )
    assert commitment == _HASH_V2
    assert attestation.commit_canonical(_CANON_V2, _SALT_E) == _HASH_V2
    assert attestation.verify_canonical(_CANON_V2, _SALT_E, _HASH_V2.upper()) is True


def test_golden_hash_v2_without_council_recipe_or_confidence() -> None:
    payload = _payload_v2(
        predicted_target_price=None,
        confidence_score=None,
        council_recommendation=None,
        council_consensus_strength=None,
        recipe_version=None,
        protocol_sha256=None,
    )
    canonical = attestation.canonicalize(payload, schema_version=attestation.SCHEMA_VERSION)
    assert canonical == _CANON_V2_BARE
    assert attestation.commit_canonical(canonical, _SALT_E) == _HASH_V2_BARE


def test_confidence_is_a_quantized_decimal_string_not_a_float() -> None:
    # Float w kanonizacji to binarny szum (0.1 + 0.2) i zależność od repr —
    # zawsze string Decimal po quantize do czterech miejsc, połówki od zera.
    payload = _payload_v2(confidence_score=0.1 + 0.2, council_consensus_strength=0.00005)
    assert payload["confidence_score"] == "0.3000"
    assert payload["council"] == {"final_recommendation": "BUY", "consensus_strength": "0.0001"}


def test_verify_canonical_rejects_a_tampered_confidence() -> None:
    tampered = _CANON_V2.replace('"confidence_score":"0.8500"', '"confidence_score":"0.9500"')
    assert attestation.verify_canonical(tampered, _SALT_E, _HASH_V2) is False


# --- WIA-01: commitment zapisany w wierszu tej samej godziny UTC ---


def test_stored_commitment_is_not_superseded_by_its_own_plaintext() -> None:
    # Retry tej samej predykcji: ta sama sól da ten sam, już opublikowany hash.
    stored = attestation.StoredCommitment(commitment=_HASH_1, salt=_SALT_A, payload=_CANON_1)
    assert stored.is_superseded_by(_CANON_1) is False


def test_stored_commitment_is_superseded_by_another_plaintext() -> None:
    stored = attestation.StoredCommitment(commitment=_HASH_1, salt=_SALT_A, payload=_CANON_1)
    assert stored.is_superseded_by(_CANON_1.replace('"up"', '"down"')) is True


def test_superseded_check_compares_hashes_not_the_stored_payload() -> None:
    # Wiersz bez `commitment_payload` (sprzed 043) też da się porównać: hash + sól.
    stored = attestation.StoredCommitment(commitment=_HASH_1, salt=_SALT_A, payload=None)
    assert stored.is_superseded_by(_CANON_1) is False


def test_superseded_reveal_status_is_frozen() -> None:
    # Status trafia do publicznego ledgera revealów — weryfikator filtruje po nim.
    assert attestation.REVEAL_STATUS_SUPERSEDED == "superseded"


# --- WIA-25: linia ledgera commitmentów --------------------------------------

_LINE_HASH = "ab" * 32


def test_ledger_record_binds_symbol_hash_prediction_time_and_schema() -> None:
    record = attestation.commitment_ledger_record(
        symbol="AAPL",
        commitment=_LINE_HASH,
        prediction_id="pred-1",
        committed_at=datetime(2026, 10, 26, 5, 31, 7, tzinfo=UTC),
    )

    assert record == {
        "symbol": "AAPL",
        "commitment": _LINE_HASH,
        "prediction_id": "pred-1",
        "committed_at": "2026-10-26T05:31:07+00:00",
        "schema_version": attestation.SCHEMA_VERSION,
    }


def test_ledger_record_states_the_time_in_utc() -> None:
    warsaw = timezone(timedelta(hours=2))

    record = attestation.commitment_ledger_record(
        symbol="AAPL",
        commitment=_LINE_HASH,
        prediction_id="pred-1",
        committed_at=datetime(2026, 10, 26, 7, 31, 7, tzinfo=warsaw),
    )

    assert record["committed_at"] == "2026-10-26T05:31:07+00:00"


def test_ledger_record_rejects_a_time_without_a_timezone() -> None:
    with pytest.raises(ValueError, match="strefy czasowej"):
        attestation.commitment_ledger_record(
            symbol="AAPL",
            commitment=_LINE_HASH,
            prediction_id="pred-1",
            committed_at=datetime(2026, 10, 26, 5, 31, 7),
        )


def test_ledger_record_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        attestation.commitment_ledger_record(
            symbol="AAPL",
            commitment=_LINE_HASH,
            prediction_id="pred-1",
            committed_at=datetime(2026, 10, 26, tzinfo=UTC),
            schema_version=99,
        )


def test_ledger_line_fields_do_not_change_the_canonicalisation() -> None:
    # Linia ledgera nie wchodzi do hasha, więc dopisanie pól nie podbija
    # SCHEMA_VERSION: golden v2 zostaje bajt w bajt (test_golden_hash_v2).
    assert attestation.SCHEMA_VERSION == 2


# --- WIA-60: drzewo Merkle'a cyklu (RFC 9162, §2.1) ---------------------------------
# Wartości policzone osobnym skryptem wprost ze wzorów RFC: liść = SHA-256(0x00 ‖ d),
# węzeł = SHA-256(0x01 ‖ lewy ‖ prawy), podział na największej potędze 2 < n.
_LEAF_A, _LEAF_B, _LEAF_C = "aa" * 32, "bb" * 32, "cc" * 32
_GOLDEN_ROOT_ABC = "d73cea60f2ce124730a688b2456b84beb6fed822ab3e6d23c30c1ccdcbb5433b"
_EMPTY_ROOT = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_GOLDEN_PROOF_C = ("03938e2c8f758e6cae443d499b41c899c373eb0c0198bae61796a069f2b05904",)
_GOLDEN_PROOF_A = (
    "4f16119d36ccd0da91102f57692d73934fd0ad2494280df88449accedbbfb7ea",
    "2e3aa189e1f666b2c3e864e21d978388020b89a6725e31ff2657bad5840a7f02",
)


def _leaves(n: int) -> list[str]:
    return [f"{index:064x}" for index in range(n)]


def test_golden_root_for_three_known_leaves() -> None:
    assert attestation.merkle_root([_LEAF_A, _LEAF_B, _LEAF_C]) == _GOLDEN_ROOT_ABC


def test_golden_inclusion_proofs_for_three_known_leaves() -> None:
    leaves = [_LEAF_A, _LEAF_B, _LEAF_C]

    assert attestation.inclusion_proof(leaves, 0) == _GOLDEN_PROOF_A
    assert attestation.inclusion_proof(leaves, 2) == _GOLDEN_PROOF_C


def test_root_of_an_empty_cycle_is_the_hash_of_nothing() -> None:
    assert attestation.merkle_root([]) == _EMPTY_ROOT


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_merkle_inclusion_proof_verifies_for_each_leaf(size: int) -> None:
    leaves = _leaves(size)
    root = attestation.merkle_root(leaves)

    for index, leaf in enumerate(leaves):
        proof = attestation.inclusion_proof(leaves, index)
        assert attestation.verify_inclusion(leaf, index, size, proof, root), (size, index)


def test_proof_fails_for_leaf_not_in_tree() -> None:
    leaves = _leaves(5)
    root = attestation.merkle_root(leaves)
    proof = attestation.inclusion_proof(leaves, 3)

    assert not attestation.verify_inclusion("ff" * 32, 3, 5, proof, root)


@pytest.mark.parametrize(
    ("index", "size", "proof_change"),
    [
        (2, 5, "keep"),  # dowód liścia 3 podany dla pozycji 2
        (3, 4, "keep"),  # rozmiar drzewa o innym kształcie ścieżki
        (3, 5, "drop"),  # obcięta ścieżka
        (3, 5, "extra"),  # ścieżka dłuższa niż drzewo
        (5, 5, "keep"),  # indeks poza drzewem
        (-1, 5, "keep"),
    ],
)
def test_proof_fails_for_a_wrong_position_size_or_path(
    index: int, size: int, proof_change: str
) -> None:
    leaves = _leaves(5)
    root = attestation.merkle_root(leaves)
    proof = list(attestation.inclusion_proof(leaves, 3))
    if proof_change == "drop":
        proof.pop()
    elif proof_change == "extra":
        proof.append("00" * 32)

    assert not attestation.verify_inclusion(leaves[3], index, size, proof, root)


def test_proof_check_rejects_text_that_is_not_hex() -> None:
    leaves = _leaves(2)
    root = attestation.merkle_root(leaves)

    assert not attestation.verify_inclusion("zz", 0, 2, ["00" * 32], root)
    assert not attestation.verify_inclusion(leaves[0], 0, 2, ["not hex"], root)


def test_merkle_input_must_be_hex() -> None:
    with pytest.raises(ValueError, match="hex"):
        attestation.merkle_root(["not a hash"])
    with pytest.raises(IndexError):
        attestation.inclusion_proof(_leaves(2), 2)


def test_cycle_root_record_is_one_line_with_sorted_unique_leaves() -> None:
    record = attestation.cycle_root_record(
        cycle_id="123.1.20261026T053000Z",
        committed_at=datetime(2026, 10, 26, 7, 31, tzinfo=timezone(timedelta(hours=2))),
        commitments=[_LEAF_C, _LEAF_A.upper(), _LEAF_B, _LEAF_C],
    )

    assert record == {
        "cycle_id": "123.1.20261026T053000Z",
        "committed_at": "2026-10-26T05:31:00+00:00",
        "root": _GOLDEN_ROOT_ABC,
        "n_leaves": 3,
        "leaves": [_LEAF_A, _LEAF_B, _LEAF_C],
        "schema_version": attestation.SCHEMA_VERSION,
        "tree": "rfc9162-sha256",
    }


def test_cycle_root_record_is_published_for_an_empty_cycle() -> None:
    record = attestation.cycle_root_record(
        cycle_id="c", committed_at=datetime(2026, 10, 26, tzinfo=UTC), commitments=[]
    )

    assert (record["root"], record["n_leaves"], record["leaves"]) == (_EMPTY_ROOT, 0, [])


def test_cycle_root_record_rejects_a_time_without_a_timezone() -> None:
    with pytest.raises(ValueError, match="strefy"):
        attestation.cycle_root_record(
            cycle_id="c", committed_at=datetime(2026, 10, 26), commitments=[]
        )


def _root_line(cycle_id: str, leaves: list[str]) -> dict[str, Any]:
    return attestation.cycle_root_record(
        cycle_id=cycle_id, committed_at=datetime(2026, 10, 26, tzinfo=UTC), commitments=leaves
    )


def test_ledger_gives_a_verifiable_inclusion_for_a_leaf_of_a_cycle() -> None:
    ledger = attestation.CommitmentLedger.from_records(
        [{"symbol": "OLD", "commitment": "dd" * 32}, _root_line("c1", [_LEAF_A, _LEAF_B, _LEAF_C])]
    )

    inclusion = ledger.inclusion_of(_LEAF_C.upper())

    assert inclusion is not None
    assert inclusion.as_record() == {
        "cycle_id": "c1",
        "root": _GOLDEN_ROOT_ABC,
        "leaf_index": 2,
        "tree_size": 3,
        "proof": list(_GOLDEN_PROOF_C),
    }
    assert inclusion.verifies(_LEAF_C)


def test_ledger_knows_legacy_single_lines_and_cycle_leaves() -> None:
    ledger = attestation.CommitmentLedger.from_records(
        [
            {"symbol": "OLD", "commitment": "DD" * 32},
            {"note": "linia bez hasha nic nie publikuje"},
            _root_line("c1", [_LEAF_A]),
        ]
    )

    assert ledger.is_published("dd" * 32)
    assert ledger.is_published(_LEAF_A)
    assert ledger.inclusion_of("dd" * 32) is None
    assert not ledger.is_published(_LEAF_B)
    assert ledger.published == frozenset({"dd" * 32, _LEAF_A})


def test_cycle_root_line_states_whether_its_root_matches_its_leaves() -> None:
    honest = attestation.CycleRoot.from_record(_root_line("c1", [_LEAF_A, _LEAF_B]))
    forged = attestation.CycleRoot.from_record(
        {**_root_line("c1", [_LEAF_A, _LEAF_B]), "leaves": [_LEAF_A]}
    )
    miscounted = attestation.CycleRoot.from_record(
        {**_root_line("c1", [_LEAF_A, _LEAF_B]), "n_leaves": 3}
    )

    garbled = attestation.CycleRoot.from_record({**_root_line("c1", [_LEAF_A]), "leaves": ["zz"]})

    assert honest is not None and honest.consistent
    assert forged is not None and not forged.consistent
    assert miscounted is not None and not miscounted.consistent
    assert garbled is not None and not garbled.consistent


@pytest.mark.parametrize(
    "record",
    [
        {"symbol": "OLD", "commitment": "dd" * 32},
        {"cycle_id": "c", "root": 5, "leaves": []},
        {"cycle_id": "c", "root": "ab", "leaves": "aa"},
        {"cycle_id": "c", "root": "ab", "leaves": [1]},
        {"cycle_id": "c", "root": "ab", "leaves": [], "n_leaves": "0"},
        {"cycle_id": None, "root": "ab", "leaves": []},
    ],
)
def test_lines_that_are_not_a_well_formed_cycle_root_are_not_roots(
    record: dict[str, Any],
) -> None:
    assert attestation.CycleRoot.from_record(record) is None


def test_incomplete_cycle_reveal_status_is_frozen() -> None:
    # Publiczny kontrakt ledgera: zmiana napisu rozjechałaby weryfikator z historią.
    assert attestation.REVEAL_STATUS_INCOMPLETE_CYCLE == "incomplete_cycle"


# WIA-73: artefakt wpisu `hashedrekord` to 32 bajty rootu — reguła zamrożona golden testem,
# bo weryfikator liczy ten sam hash z linii ledgera latami później.
_GOLDEN_ROOT_ABC_ARTIFACT_SHA256 = (
    "ff56b960c3d1e1f581d065b766ba775befbdcc53451a0ced46d7b011aeea6761"
)


def test_golden_rekor_artifact_hash_of_a_known_root() -> None:
    assert attestation.root_artifact(_GOLDEN_ROOT_ABC) == bytes.fromhex(_GOLDEN_ROOT_ABC)
    assert attestation.root_artifact_sha256(_GOLDEN_ROOT_ABC) == _GOLDEN_ROOT_ABC_ARTIFACT_SHA256


@pytest.mark.parametrize("root", ["ab" * 31, "ab" * 33, "zz" * 32])
def test_rekor_artifact_is_exactly_the_32_bytes_of_a_root(root: str) -> None:
    with pytest.raises(ValueError):
        attestation.root_artifact(root)
