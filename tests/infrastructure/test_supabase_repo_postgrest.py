"""TEST-17: `SupabaseRepository` na prawdziwym PostgREST w kontenerze.

Adapter testowano mockami łańcucha `table().select()…`, a schemat osobno
(test_migrations.py), więc szew adapter ↔ schemat nie miał testu: PGRST204 przy
kolumnie bez migracji (incydent 013), NUMERIC wracający jako liczba JSON, cicha
granica ~1000 wierszy (G48, #8). Tu jeden zestaw kontenerów na moduł (pamięć
maszyny): Postgres z pgvector z kompletem migracji i rolami jak w Supabase oraz
`postgrest/postgrest` z JWT podpisanym lokalnym sekretem — repozytorium mówi do
niego tym samym klientem `supabase-py` co na produkcji.

Jedyna różnica wobec Supabase: brak bramy, która wystawia PostgREST pod `/rest/v1`
— klient dostaje adres PostgREST bez prefiksu (`_root_rest_client`). Brak też
schematu `auth`: migracje go nie używają.

Uruchomienie: `uv run pytest -m containers tests/infrastructure/test_supabase_repo_postgrest.py`
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest import mock

import psycopg2
import pytest
import requests
from postgrest.exceptions import APIError
from supabase import Client, create_client
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.postgres import PostgresContainer

from src.application.errors import NotFound, PortError
from src.infrastructure.adapters import supabase_repo
from src.infrastructure.adapters.supabase_repo import SupabaseRepository
from tests.infrastructure.test_migrations import (
    PGVECTOR_IMAGE,
    REQUIRE_DOCKER_ENV,
    SUPABASE_API_ROLES,
    SUPABASE_DEFAULT_GRANTS,
    _apply_migrations,
)

pytestmark = pytest.mark.containers

# Wersja przypięta (wydana dawno przed dodaniem testu); hosted Supabase stoi na 12.x.
POSTGREST_IMAGE = "postgrest/postgrest:v12.2.3"
POSTGREST_PORT = 3000
# Lokalny sekret JWT tylko dla tego kontenera — nie ma nic wspólnego z produkcją.
JWT_SECRET = "stockagent-test17-local-jwt-secret-0123456789"
AUTHENTICATOR_PASSWORD = "authenticator-test17"
DB_ALIAS = "db"
# Hosted Supabase tnie SELECT do 1000 wierszy bez błędu (#8) — ten sam limit tutaj.
POSTGREST_MAX_ROWS = 1000
_READY_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class PostgrestStack:
    base_url: str
    conn: psycopg2.extensions.connection


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def jwt_for(role: str) -> str:
    """JWT HS256 z claimem `role` — tak klucz anon / service_role wybiera rolę bazy."""
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({"role": role, "iss": "stockagent-test17"}).encode())
    signing_input = f"{header}.{payload}".encode()
    signature = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(signature)}"


def _require_docker() -> None:
    """Bez demona skip, chyba że `REQUIRE_DOCKER=1` (krok `containers` w CI)."""
    import docker
    from docker.errors import DockerException

    try:
        docker.from_env().ping()
    except DockerException as exc:
        if os.environ.get(REQUIRE_DOCKER_ENV) == "1":
            pytest.fail(f"Docker daemon niedostępny, a {REQUIRE_DOCKER_ENV}=1: {exc}")
        pytest.skip("Docker daemon niedostępny — pomijam testy kontenerowe")


def _prepare_supabase_database(conn: psycopg2.extensions.connection) -> None:
    """Role API i domyślne granty jak w Supabase, `authenticator` dla PostgREST,
    potem komplet migracji — kolejność jak na produkcji."""
    with conn.cursor() as cur:
        for role, attributes in SUPABASE_API_ROLES.items():
            cur.execute(f"CREATE ROLE {role} {attributes}")
        cur.execute(
            "CREATE ROLE authenticator LOGIN NOINHERIT PASSWORD %s", (AUTHENTICATOR_PASSWORD,)
        )
        cur.execute(f"GRANT {', '.join(SUPABASE_API_ROLES)} TO authenticator")
        cur.execute(SUPABASE_DEFAULT_GRANTS)
    _apply_migrations(conn)


def _wait_until_ready(base_url: str) -> None:
    """PostgREST odpowiada 503, dopóki nie wczyta cache'u schematu."""
    headers: dict[str, str | bytes] = {"Authorization": f"Bearer {jwt_for('service_role')}"}
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while True:
        try:
            response = requests.get(
                f"{base_url}/prediction_logs", params={"limit": "0"}, headers=headers, timeout=5
            )
            if response.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        if time.monotonic() > deadline:
            pytest.fail(f"PostgREST nie wstał w {_READY_TIMEOUT_S:.0f} s")
        time.sleep(0.5)


@pytest.fixture(scope="module")
def postgrest_stack() -> Iterator[PostgrestStack]:
    _require_docker()
    with (
        Network() as network,
        PostgresContainer(PGVECTOR_IMAGE)
        .with_network(network)
        .with_network_aliases(DB_ALIAS) as postgres,
    ):
        conn = psycopg2.connect(
            host=postgres.get_container_host_ip(),
            port=postgres.get_exposed_port(5432),
            dbname=postgres.dbname,
            user=postgres.username,
            password=postgres.password,
        )
        conn.autocommit = True
        _prepare_supabase_database(conn)
        postgrest = (
            DockerContainer(POSTGREST_IMAGE)
            .with_network(network)
            .with_env(
                "PGRST_DB_URI",
                f"postgres://authenticator:{AUTHENTICATOR_PASSWORD}@{DB_ALIAS}:5432/"
                f"{postgres.dbname}",
            )
            .with_env("PGRST_DB_SCHEMAS", "public")
            .with_env("PGRST_DB_ANON_ROLE", "anon")
            .with_env("PGRST_JWT_SECRET", JWT_SECRET)
            .with_env("PGRST_DB_MAX_ROWS", str(POSTGREST_MAX_ROWS))
            .with_exposed_ports(POSTGREST_PORT)
        )
        try:
            with postgrest:
                base_url = (
                    f"http://{postgrest.get_container_host_ip()}:"
                    f"{postgrest.get_exposed_port(POSTGREST_PORT)}"
                )
                _wait_until_ready(base_url)
                yield PostgrestStack(base_url=base_url, conn=conn)
        finally:
            conn.close()


def _root_rest_client(url: str, key: str) -> Client:
    """Klient supabase-py z REST pod korzeniem PostgREST zamiast `/rest/v1`."""
    client = create_client(url, key)
    client.rest_url = client.supabase_url
    return client


def make_repository(stack: PostgrestStack, role: str = "service_role") -> SupabaseRepository:
    with mock.patch.object(supabase_repo, "create_client", _root_rest_client):
        return SupabaseRepository(url=stack.base_url, key=jwt_for(role))


def prediction_record(symbol: str, **overrides: Any) -> dict[str, Any]:
    """Rekord jak z `save_node` (agent_graph.py) — ceny jako Decimal."""
    record: dict[str, Any] = {
        "symbol": symbol,
        "price_at_prediction": Decimal("187.1235"),
        "sentiment_score": 0.21,
        "av_relevance_avg": 0.74,
        "news_volume_24h": 3,
        "high_relevance_count": 1,
        "av_llm_agreement": 0.5,
        "news_summary": "Redacted news summary.",
        "predicted_trend": "BULLISH",
        "predicted_target_price": Decimal("190.5"),
        "confidence_score": 0.62,
        "reasoning_text": "Test reasoning.",
    }
    record.update(overrides)
    return record


@pytest.fixture(scope="module")
def repository(postgrest_stack: PostgrestStack) -> SupabaseRepository:
    return make_repository(postgrest_stack)


class TestPredictionRoundTrip:
    def test_save_prediction_round_trips_through_postgrest(
        self, repository: SupabaseRepository
    ) -> None:
        saved_id = repository.save_prediction(prediction_record("T17RT"))

        prediction = repository.get_unverified_prediction("T17RT")

        assert prediction is not None
        assert prediction.id == saved_id
        # NUMERIC wraca z PostgREST jako liczba JSON — kontrakt portu to Decimal
        # w skali kolumny DECIMAL(14,4), bez śladu binarnego floata.
        assert isinstance(prediction.price_at_prediction, Decimal)
        assert prediction.price_at_prediction == Decimal("187.1235")
        assert prediction.predicted_target_price == Decimal("190.5")
        assert prediction.predicted_trend == "BULLISH"
        assert prediction.news_summary == "Redacted news summary."
        assert prediction.created_at is not None
        assert prediction.created_at.tzinfo is not None


def _stored_target_prices(conn: psycopg2.extensions.connection, symbol: str) -> list[Decimal]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT predicted_target_price FROM prediction_logs WHERE symbol = %s "
            "ORDER BY timestamp",
            (symbol,),
        )
        return [row[0] for row in cur.fetchall()]


def _hours_ago(hours: int) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


class TestIdempotentUpsert:
    def test_same_hour_save_updates_the_one_row(
        self, repository: SupabaseRepository, postgrest_stack: PostgrestStack
    ) -> None:
        first = repository.save_prediction(
            prediction_record("T17UP", timestamp="2026-10-20T10:05:00+00:00")
        )
        second = repository.save_prediction(
            prediction_record(
                "T17UP",
                timestamp="2026-10-20T10:50:00+00:00",
                predicted_target_price=Decimal("191.25"),
            )
        )

        assert second == first
        assert _stored_target_prices(postgrest_stack.conn, "T17UP") == [Decimal("191.2500")]

    def test_next_hour_save_adds_a_row(
        self, repository: SupabaseRepository, postgrest_stack: PostgrestStack
    ) -> None:
        first = repository.save_prediction(
            prediction_record("T17NH", timestamp="2026-10-20T10:05:00+00:00")
        )
        second = repository.save_prediction(
            prediction_record("T17NH", timestamp="2026-10-20T11:05:00+00:00")
        )

        assert second != first
        assert len(_stored_target_prices(postgrest_stack.conn, "T17NH")) == 2

    def test_price_is_stored_in_the_column_scale(self, repository: SupabaseRepository) -> None:
        # ARCH-38: adapter zaokrągla do DECIMAL(14,4) jak Postgres — odczyt oddaje
        # dokładnie to, co baza przechowała.
        repository.save_prediction(
            prediction_record("T17SC", price_at_prediction=Decimal("65432.123456"))
        )

        prediction = repository.get_unverified_prediction("T17SC")

        assert prediction is not None
        assert prediction.price_at_prediction == Decimal("65432.1235")


class TestUnverifiedQueue:
    def test_oldest_unverified_past_min_age_comes_first(
        self, repository: SupabaseRepository
    ) -> None:
        # Filtry czasu idą do PostgREST jako ISO z „+00:00” — `+` w query stringu
        # bez kodowania byłby spacją i filtr padałby po cichu.
        oldest = repository.save_prediction(prediction_record("T17Q", timestamp=_hours_ago(40)))
        repository.save_prediction(prediction_record("T17Q", timestamp=_hours_ago(20)))
        repository.save_prediction(prediction_record("T17Q"))

        prediction = repository.get_unverified_prediction("T17Q", min_age_hours=12)

        assert prediction is not None
        assert prediction.id == oldest

    def test_max_age_skips_predictions_older_than_the_window(
        self, repository: SupabaseRepository
    ) -> None:
        repository.save_prediction(prediction_record("T17W", timestamp=_hours_ago(100)))
        in_window = repository.save_prediction(prediction_record("T17W", timestamp=_hours_ago(30)))

        prediction = repository.get_unverified_prediction(
            "T17W", min_age_hours=12, max_age_hours=48
        )

        assert prediction is not None
        assert prediction.id == in_window

    def test_settled_prediction_leaves_the_queue(
        self, repository: SupabaseRepository, postgrest_stack: PostgrestStack
    ) -> None:
        settled = repository.save_prediction(prediction_record("T17ST", timestamp=_hours_ago(30)))
        with postgrest_stack.conn.cursor() as cur:
            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 188 WHERE id = %s", (settled,)
            )

        assert repository.get_unverified_prediction("T17ST") is None


# Wymiar kolumny `embedding` (migracja 003, text-embedding-3-small).
EMBEDDING_DIM = 1536


def _unit_vector(axis: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[axis] = 1.0
    return vector


class TestSimilaritySearch:
    """RPC `match_news_embeddings` (011, 017). Adapter połyka każdy błąd RPC i oddaje
    `[]`, więc zła nazwa parametru albo typ wektora wyglądałyby jak pusta pamięć."""

    def test_find_similar_predictions_ranks_the_nearest_first(
        self, repository: SupabaseRepository
    ) -> None:
        nearest = repository.save_prediction(prediction_record("T17S1", embedding=_unit_vector(7)))
        repository.save_prediction(prediction_record("T17S2", embedding=_unit_vector(8)))

        results = repository.find_similar_predictions(_unit_vector(7), limit=2)

        assert results, "RPC zwróciło pustą listę — błąd połknięty przez adapter"
        assert results[0]["id"] == nearest
        assert results[0]["similarity"] == pytest.approx(1.0)

    def test_regime_filter_drops_analogues_from_another_regime(
        self, repository: SupabaseRepository
    ) -> None:
        same = repository.save_prediction(
            prediction_record("T17R1", embedding=_unit_vector(11), regime="RISK_OFF")
        )
        other = repository.save_prediction(
            prediction_record("T17R2", embedding=_unit_vector(11), regime="RISK_ON")
        )

        results = repository.find_similar_predictions(_unit_vector(11), limit=10, regime="risk off")

        ids = [row["id"] for row in results]
        assert same in ids
        assert other not in ids


class TestPagination:
    def test_price_history_reads_past_the_postgrest_row_cap(
        self, repository: SupabaseRepository, postgrest_stack: PostgrestStack
    ) -> None:
        total = POSTGREST_MAX_ROWS + 205
        with postgrest_stack.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) "
                "SELECT 'T17PG', 100 + g, now() - g * interval '1 hour' "
                "FROM generate_series(1, %s) AS g",
                (total,),
            )
        capped = requests.get(
            f"{postgrest_stack.base_url}/price_snapshots",
            params={"symbol": "eq.T17PG", "select": "id"},
            headers={"Authorization": f"Bearer {jwt_for('service_role')}"},
            timeout=10,
        )

        history = repository.get_price_history("T17PG", days=total // 24 + 2)

        # Strażnik emulacji: zwykły SELECT naprawdę jest ucięty jak w hosted Supabase.
        assert len(capped.json()) == POSTGREST_MAX_ROWS
        assert len(history) == total
        stamps = [stamp for stamp, _ in history]
        assert stamps == sorted(stamps)


class TestLlmCostLedger:
    def test_llm_cost_since_sums_priced_rows_of_the_window_past_the_row_cap(
        self, repository: SupabaseRepository, postgrest_stack: PostgrestStack
    ) -> None:
        # FIN-25: suma okna limitu z wierszy `llm_calls` (040) — także ponad limit
        # 1000 wierszy PostgREST; wiersz bez wyceny i wiersz sprzed okna poza sumą.
        since = datetime(2026, 10, 25, tzinfo=UTC)
        in_window = POSTGREST_MAX_ROWS + 205
        with postgrest_stack.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO llm_calls (run_id, node, provider, model, input_tokens, "
                "output_tokens, cost_usd, created_at) "
                "SELECT 't17-fin25', 'predict', 'anthropic', 'm', 1, 1, 0.002, "
                "%s::timestamptz + g * interval '1 second' "
                "FROM generate_series(1, %s) AS g",
                (since.isoformat(), in_window),
            )
            cur.execute(
                "INSERT INTO llm_calls (run_id, node, provider, model, input_tokens, "
                "output_tokens, cost_usd, created_at) VALUES "
                "('t17-fin25', 'predict', 'anthropic', 'm', 1, 1, NULL, %s), "
                "('t17-fin25', 'predict', 'anthropic', 'm', 1, 1, 5.0, %s)",
                (
                    (since + timedelta(hours=1)).isoformat(),
                    (since - timedelta(seconds=1)).isoformat(),
                ),
            )

        spent = repository.get_llm_cost_since(since)

        assert spent == pytest.approx(in_window * 0.002)


class TestSchemaAndRoles:
    def test_unknown_column_is_rejected_as_not_found(
        self, repository: SupabaseRepository
    ) -> None:
        # Incydent 013: klucz bez migracji → PGRST204 i cały zapis predykcji przepada.
        with pytest.raises(NotFound) as excinfo:
            repository.save_prediction(prediction_record("T17NC", not_a_column=1))

        assert isinstance(excinfo.value, APIError)
        assert excinfo.value.code == "PGRST204"

    def test_anon_key_cannot_write_predictions(
        self, postgrest_stack: PostgrestStack
    ) -> None:
        # 038: RLS bez polityk i granty odebrane rolom publicznym — klucz anon
        # nie zapisze ani nie przeczyta niczego.
        anonymous = make_repository(postgrest_stack, role="anon")

        with pytest.raises(PortError):
            anonymous.save_prediction(prediction_record("T17AN"))

        assert _stored_target_prices(postgrest_stack.conn, "T17AN") == []
