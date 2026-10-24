"""Testy migracji SQL z prawdziwym Postgres + pgvector (Docker).

Uruchamia kontener `pgvector/pgvector:pg16`, aplikuje wszystkie migracje
w kolejności i weryfikuje schemat — w szczególności kolumnę `embedding`,
której brak powodował błędy PGRST204 na produkcji.

Uruchomienie:
    uv run pytest -m containers
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql
from testcontainers.postgres import PostgresContainer

from src.domain.calibration_curve import brier_score, log_loss
from src.domain.persona_track_record import (
    PersonaVoteOutcome,
    persona_accuracy_weights_as_of,
    persona_hit_stats,
    persona_vote_correct,
)
from src.domain.value_objects import AssetType

# `supabase db push` czyta WYŁĄCZNIE `supabase/migrations/` — trzymamy je tam,
# żeby CLI i testy kontenerowe widziały dokładnie ten sam zbiór plików.
MIGRATIONS_DIR = Path(__file__).parents[2] / "supabase" / "migrations"
MIGRATION_FILES = [
    "001_init.sql",
    "002_price_snapshots.sql",
    "003_add_embedding.sql",
    "004_align_ml_feature_store.sql",
    "005_council_verdict.sql",
    "006_fundamentals_cache.sql",
    "007_council_votes.sql",
    "008_data_quality_flags.sql",
    "009_trend_correctness.sql",
    "010_quota_alerts.sql",
    "011_match_news_embeddings.sql",
    "012_ml_feature_store_return_target.sql",
    "013_idempotency_and_pagination.sql",
    "014_subscribers.sql",
    "015_persona_accuracy.sql",
    "016_confidence_calibration.sql",
    "017_vector_memory_regime.sql",
    "018_persona_track_record.sql",
    "019_model_scorecards.sql",
    "020_decision_receipts.sql",
    "021_shock_alerts.sql",
    "022_alpha_fusion_score.sql",
    "023_positions.sql",
    "024_attestation.sql",
    "025_implied_edge.sql",
    "026_feature_store_clean_rows.sql",
    "027_feature_set_version.sql",
    "028_model_scorecard_baselines.sql",
    "029_prediction_settled_at.sql",
    "030_recipe_version.sql",
    "031_feature_snapshots.sql",
    "032_move_interval.sql",
    "033_regime_history.sql",
    "034_gate_delta.sql",
    "035_positions_currency.sql",
    "036_fx_rates.sql",
    "037_experiment_arms.sql",
    "038_enable_rls.sql",
    "039_feature_store_av_degraded_rows.sql",
    "040_llm_calls.sql",
    "041_cycle_runs.sql",
    "042_reference_predictions.sql",
    "043_attestation_payload.sql",
    "044_calibration_score_version.sql",
    "045_persona_vote_outcomes.sql",
    "046_correction_category.sql",
    "047_ml_feature_store_lag_over_all_predictions.sql",
    "048_normalize_predicted_trend.sql",
    "049_model_scorecards_run_id.sql",
    "050_feature_store_label_horizon.sql",
    "051_trend_probabilities.sql",
    "052_model_scorecards_feature_stats.sql",
    "053_earnings_cache.sql",
    "054_council_votes_unique.sql",
    "055_prediction_stale_quote.sql",
    "056_fundamentals_market_cap.sql",
    "057_price_level_alerts.sql",
    "058_settled_reads_without_stale_quote.sql",
    "059_settlement_session_close.sql",
    "060_model_scorecards_weights_sha256.sql",
    "061_digests.sql",
    "062_corporate_actions.sql",
    "063_rag_exact_search.sql",
    "064_model_scorecards_provenance.sql",
    "065_persona_scoring.sql",
    "070_rpc_search_path.sql",
    "071_schema_version.sql",
]

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
# CI-06: krok `containers` w ci.yml ustawia 1 — brak demona to wtedy porażka, nie skip.
REQUIRE_DOCKER_ENV = "REQUIRE_DOCKER"


@contextmanager
def _pgvector_connection() -> Iterator[psycopg2.extensions.connection]:
    """Kontener Postgres z pgvector i połączenie z autocommit — wspólne dla
    `pg_conn` i `fresh_pg`.

    Bez demona Dockera skipuje (lokalnie bez Docker Desktop), chyba że
    `REQUIRE_DOCKER=1`: zielony krok CI z zerem wykonanych testów niczego nie dowodzi.
    """
    import docker

    try:
        docker.from_env().ping()
    except Exception as exc:
        if os.environ.get(REQUIRE_DOCKER_ENV) == "1":
            pytest.fail(f"Docker daemon niedostępny, a {REQUIRE_DOCKER_ENV}=1: {exc}")
        pytest.skip("Docker daemon niedostępny — pomijam testy kontenerowe")

    with PostgresContainer(PGVECTOR_IMAGE) as container:
        conn = psycopg2.connect(
            host=container.get_container_host_ip(),
            port=container.get_exposed_port(5432),
            dbname=container.dbname,
            user=container.username,
            password=container.password,
        )
        conn.autocommit = True
        try:
            yield conn
        finally:
            conn.close()


@pytest.fixture(scope="module")
def pg_conn():
    """Kontener Postgres z pgvector — żyje przez cały moduł testów.

    Skipuje cały moduł gdy Docker nie jest dostępny (CI bez demona, lokalnie
    bez Docker Desktop) — poza `REQUIRE_DOCKER=1` (`_pgvector_connection`).
    """
    with _pgvector_connection() as conn:
        yield conn


def _apply_migrations(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        for filename in MIGRATION_FILES:
            sql = (MIGRATIONS_DIR / filename).read_text()
            cur.execute(sql)


MIGRATED_DB = "migrated_schema"


@pytest.fixture(scope="module")
def migrated_pg_conn(
    pg_conn: psycopg2.extensions.connection,
) -> Iterator[psycopg2.extensions.connection]:
    """TEST-08 — osobna baza w kontenerze `pg_conn` z kompletem migracji, które
    aplikuje SAM fixture.

    `pg_conn` zostaje surowy: schemat dostaje dopiero
    `TestMigrations::test_migrations_apply_without_errors`, więc testy schematu
    na `pg_conn` zależą od kolejności i uruchomione same (`-k`) padają. Nowy test,
    któremu wystarcza gotowy schemat, bierze ten fixture — przechodzi w izolacji
    i w dowolnej kolejności. Testy samego aplikowania migracji zostają na
    `pg_conn`, a stan sprzed konkretnej migracji daje `fresh_pg`.

    Baza jest wspólna dla całego modułu: test wstawiający wiersze używa
    unikalnego symbolu albo `_rolled_back`. Druga baza w tym samym kontenerze,
    nie drugi kontener — kolejne setki MB przy pełnym swapie."""
    with pg_conn.cursor() as cur:
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(MIGRATED_DB)))
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(MIGRATED_DB)))

    info = pg_conn.info
    conn = psycopg2.connect(
        host=info.host,
        port=info.port,
        dbname=MIGRATED_DB,
        user=info.user,
        password=info.password,
    )
    conn.autocommit = True
    _apply_migrations(conn)
    yield conn
    conn.close()


def _columns(conn: psycopg2.extensions.connection, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


def _indexes(conn: psycopg2.extensions.connection, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'public' AND tablename = %s
            """,
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


# Tabele `public` aplikacji (relkind r/p, bez obiektów rozszerzeń) z flagą RLS.
_PUBLIC_TABLES_SQL = """
    SELECT c.relname, c.relrowsecurity
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p')
      AND NOT EXISTS (
          SELECT 1 FROM pg_depend d
          WHERE d.classid = 'pg_class'::regclass AND d.objid = c.oid AND d.deptype = 'e'
      )
"""


@pytest.mark.containers
class TestMigrations:
    def test_migrations_apply_without_errors(self, pg_conn):
        _apply_migrations(pg_conn)

    def test_prediction_logs_has_required_columns(self, pg_conn):
        cols = _columns(pg_conn, "prediction_logs")
        required = {
            "id", "symbol", "timestamp",
            "price_at_prediction", "sentiment_score", "news_summary",
            "av_relevance_avg", "news_volume_24h", "high_relevance_count",
            "av_llm_agreement",
            "predicted_trend", "predicted_target_price", "reasoning_text",
            "actual_price_after_12h", "accuracy_score", "correction_insights",
            "is_trend_correct",
            "embedding",
        }
        assert required <= cols, f"Brakujące kolumny: {required - cols}"

    def test_price_snapshots_table_exists(self, pg_conn):
        cols = _columns(pg_conn, "price_snapshots")
        assert {"id", "symbol", "price", "timestamp"} <= cols

    def test_ml_feature_store_exposes_fast_loop_feature_contract(self, pg_conn):
        cols = _columns(pg_conn, "ml_feature_store")
        required = {
            "price_delta",
            "av_sentiment_score",
            "av_relevance_avg",
            "news_volume_24h",
            "high_relevance_count",
            "llm_trend_signal",
            "av_llm_agreement",
            # Target to ZWROT 12h (migracja 012), nie cena bezwzględna.
            "target_return",
        }
        assert required <= cols, f"Brakujące kolumny: {required - cols}"

    def test_embedding_column_accepts_1536_dim_vector(self, pg_conn):
        with pg_conn.cursor() as cur:
            vector = "[" + ",".join(["0.1"] * 1536) + "]"
            cur.execute(
                """
                INSERT INTO prediction_logs (symbol, embedding)
                VALUES (%s, %s::vector)
                RETURNING id
                """,
                ("TEST", vector),
            )
            row = cur.fetchone()
        assert row is not None

    def test_embedding_column_is_nullable(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol) VALUES (%s) RETURNING id",
                ("TEST_NULL",),
            )
            row = cur.fetchone()
        assert row is not None

    def test_ivfflat_index_created_on_embedding(self, pg_conn):
        indexes = _indexes(pg_conn, "prediction_logs")
        assert "idx_prediction_logs_embedding" in indexes

    def test_match_news_embeddings_rpc_returns_similar_rows(self, pg_conn):
        """RPC RAG (migracja 011) zwraca historyczne predykcje wg podobieństwa
        embeddingu, z polem similarity.

        Regresja: operator pgvector `<=>` to dystans KOSINUSOWY — mierzy tylko
        kierunek wektora, nie jego długość. Wektory stałe ([0.1]*1536 vs
        [0.9]*1536) są równoległe (każdy = c·[1,...,1]), więc mają IDENTYCZNY
        (zerowy) dystans kosinusowy do dowolnego zapytania. `ORDER BY ... LIMIT
        1` rozstrzygał wtedy remis losowo i potrafił zwrócić wiersz "TEST"
        wstrzyknięty przez inny test (współdzielony, module-scoped kontener).
        Fixture'y muszą więc różnić się KIERUNKIEM, a test izolować swój zbiór
        embeddingów, żeby wynik był deterministyczny niezależnie od kolejności
        testów."""

        def embed(head: float, tail: float) -> str:
            # 1536-dim: pierwsza połowa = head, druga = tail → różny KIERUNEK.
            return "[" + ",".join([str(head)] * 768 + [str(tail)] * 768) + "]"

        with pg_conn.cursor() as cur:
            # Determinizm: ivfflat (lists=100, migracja 003) to indeks
            # PRZYBLIŻONY — przy domyślnym probes=1 skanuje tylko 1 z 100 list,
            # więc na tabeli z paroma wierszami potrafi zwrócić 0 trafień (KNN
            # mija właściwą listę). Podbijamy probes do liczby list → pełny skan
            # = dokładne KNN, niezależne od planera i liczby wierszy. Produkcyjny
            # RPC zostaje przybliżony (przy dużych danych recall jest OK) — to
            # tylko utwardzenie testu, nie zmiana zachowania RPC.
            cur.execute("SET ivfflat.probes = 100")
            # Izolacja od embeddingów wstrzykniętych przez inne testy w tym
            # samym kontenerze (np. "TEST" z testu akceptacji 1536-dim).
            cur.execute("DELETE FROM prediction_logs WHERE embedding IS NOT NULL")

            # NEAR jest kierunkowo zgodny z query, FAR — przeciwny.
            query = embed(1.0, 0.0)
            near = embed(0.9, 0.1)
            far = embed(0.1, 0.9)
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, news_summary, predicted_trend, is_trend_correct, "
                " actual_price_after_12h, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s::vector),(%s,%s,%s,%s,%s,%s::vector)",
                (
                    "NEAR", "Fed hawkish", "BEARISH", True, 100.0, near,
                    "FAR", "Earnings beat", "BULLISH", False, 200.0, far,
                ),
            )
            cur.execute(
                "SELECT symbol, similarity FROM "
                "match_news_embeddings(%s::vector, %s)",
                (query, 1),
            )
            rows = cur.fetchall()
        # Najbliższy kierunkowo jest "NEAR" (dystans ~0.006 << FAR ~0.89).
        assert len(rows) == 1
        assert rows[0][0] == "NEAR"

    def test_persona_accuracy_stats_rpc_returns_hit_rate_and_vote_count(self, pg_conn):
        """RPC leaderboardu person (#3, migracja 018) zwraca surowy hit-rate
        ORAZ liczbę rozliczonych głosów.

        Regresja kontraktu: `persona_accuracy_weights` (015) niesie tylko
        przeskalowaną wagę ∈ [0.5, 1.5], więc `vote_count` był nieodtwarzalny
        — a bez niego próg `min_votes` w domenie nie ma czego odcinać.
        Fixture: BUY na rosnącej cenie (trafny) + BUY na spadającej (chybiony)
        → hit_rate 0.5 przy 2 głosach. Predykcja nierozliczona
        (actual_price_after_12h IS NULL) NIE może wejść do próbki.
        """
        with pg_conn.cursor() as cur:
            # Izolacja: RPC agreguje po całej tabeli, a kontener jest
            # współdzielony przez moduł — czyścimy głosy innych testów.
            cur.execute("DELETE FROM council_votes")

            def add_vote(symbol, price_before, price_after, recommendation):
                cur.execute(
                    "INSERT INTO prediction_logs "
                    "(symbol, price_at_prediction, actual_price_after_12h) "
                    "VALUES (%s, %s, %s) RETURNING id",
                    (symbol, price_before, price_after),
                )
                prediction_id = cur.fetchone()[0]
                cur.execute(
                    "INSERT INTO council_votes "
                    "(prediction_id, symbol, investor_name, recommendation, confidence) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (prediction_id, symbol, "Buffett", recommendation, 0.8),
                )

            add_vote("PTR_HIT", 100.0, 110.0, "BUY")   # trafiony
            add_vote("PTR_MISS", 100.0, 90.0, "BUY")   # chybiony
            # Nierozliczona predykcja — poza próbką (actual IS NULL).
            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction) "
                "VALUES (%s, %s) RETURNING id",
                ("PTR_OPEN", 100.0),
            )
            open_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO council_votes "
                "(prediction_id, symbol, investor_name, recommendation, confidence) "
                "VALUES (%s, %s, %s, %s, %s)",
                (open_id, "PTR_OPEN", "Buffett", "BUY", 0.8),
            )

            cur.execute(
                "SELECT investor_name, hit_rate, vote_count "
                "FROM persona_accuracy_stats(%s)",
                (90,),
            )
            rows = cur.fetchall()

        assert len(rows) == 1
        name, hit_rate, vote_count = rows[0]
        assert name == "Buffett"
        assert float(hit_rate) == pytest.approx(0.5)
        assert vote_count == 2

    def test_council_verdict_column_exists(self, pg_conn):
        cols = _columns(pg_conn, "prediction_logs")
        assert "council_verdict" in cols, "Brak kolumny council_verdict w prediction_logs"

    def test_council_verdict_column_is_nullable(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol) VALUES (%s) RETURNING id",
                ("TEST_COUNCIL_NULL",),
            )
            row = cur.fetchone()
        assert row is not None

    def test_council_verdict_accepts_json(self, pg_conn):
        import json

        verdict = json.dumps(
            {
                "final_recommendation": "BUY",
                "consensus_strength": 0.75,
                "summary": "Rada kupuje.",
                "investor_opinions": [],
            }
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, council_verdict)"
                " VALUES (%s, %s::jsonb) RETURNING id",
                ("TEST_COUNCIL_JSON", verdict),
            )
            row = cur.fetchone()
        assert row is not None

    def test_is_trend_correct_column_exists(self, pg_conn):
        cols = _columns(pg_conn, "prediction_logs")
        assert "is_trend_correct" in cols, "Brak kolumny is_trend_correct"

    def test_is_trend_correct_accepts_boolean_and_null(self, pg_conn):
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, is_trend_correct)"
                " VALUES (%s, %s), (%s, %s), (%s, NULL) RETURNING id",
                ("TEST_TC_T", True, "TEST_TC_F", False, "TEST_TC_NULL"),
            )
            rows = cur.fetchall()
        assert len(rows) == 3

    def test_migrations_are_idempotent(self, pg_conn):
        # Drugie uruchomienie tych samych migracji nie może rzucić błędu
        _apply_migrations(pg_conn)

    def test_fundamentals_cache_table_has_required_columns(self, pg_conn) -> None:
        cols = _columns(pg_conn, "fundamentals_cache")
        required = {
            "symbol",
            "trailing_pe",
            "forward_pe",
            "peg_ratio",
            "eps_growth_yoy",
            "fetched_at",
        }
        assert required <= cols, f"Brakujące kolumny: {required - cols}"

    def test_fundamentals_cache_has_primary_key_on_symbol(self, pg_conn) -> None:
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = 'fundamentals_cache' AND constraint_type = 'PRIMARY KEY'
                """
            )
            constraints = [row[0] for row in cur.fetchall()]
        assert len(constraints) > 0, "Brak primary key na fundamentals_cache"

    def test_fundamentals_cache_fetched_at_index_exists(self, pg_conn) -> None:
        indexes = _indexes(pg_conn, "fundamentals_cache")
        assert "idx_fundamentals_cache_fetched_at" in indexes

    def test_subscribers_table_has_required_columns(self, pg_conn) -> None:
        cols = _columns(pg_conn, "subscribers")
        assert {"id", "email", "symbols"} <= cols, f"Brakujące kolumny: {cols}"

    # -- Migracja 013: idempotency (timestamp_hour + UNIQUE) -----------------

    def test_price_snapshots_has_timestamp_hour_column(self, pg_conn) -> None:
        cols = _columns(pg_conn, "price_snapshots")
        assert "timestamp_hour" in cols, "Brak kolumny timestamp_hour (migracja 013)"

    def test_prediction_logs_has_timestamp_hour_column(self, pg_conn) -> None:
        cols = _columns(pg_conn, "prediction_logs")
        assert "timestamp_hour" in cols, "Brak kolumny timestamp_hour (migracja 013)"

    def test_price_snapshots_has_unique_index_on_symbol_hour(self, pg_conn) -> None:
        indexes = _indexes(pg_conn, "price_snapshots")
        assert "idx_price_snapshots_symbol_hour_uniq" in indexes

    def test_prediction_logs_has_unique_index_on_symbol_hour(self, pg_conn) -> None:
        indexes = _indexes(pg_conn, "prediction_logs")
        assert "idx_prediction_logs_symbol_hour_uniq" in indexes

    def test_timestamp_hour_truncates_to_hour(self, pg_conn) -> None:
        """timestamp_hour to date_trunc('hour', timestamp) — dwa snapshoty w tej
        samej godzinie kolidują (UNIQUE), więc upsert aktualizuje, nie duplikuje."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) "
                "VALUES (%s, %s, %s) RETURNING timestamp_hour",
                ("TS_HOUR", 100.0, "2026-06-01T14:37:00+00:00"),
            )
            row = cur.fetchone()
        assert row is not None
        assert str(row[0]).startswith("2026-06-01 14:00:00")

    def test_price_snapshots_unique_constraint_rejects_same_hour_duplicate(
        self, pg_conn
    ) -> None:
        """UNIQUE(symbol, timestamp_hour): drugi INSERT w tej samej godzinie bez
        ON CONFLICT musi rzucić — to twardy fundament idempotentnego upsertu."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) "
                "VALUES (%s, %s, %s)",
                ("DUP_HOUR", 100.0, "2026-06-02T09:10:00+00:00"),
            )
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO price_snapshots (symbol, price, timestamp) "
                    "VALUES (%s, %s, %s)",
                    ("DUP_HOUR", 200.0, "2026-06-02T09:55:00+00:00"),
                )

    def test_prediction_logs_upsert_on_conflict_updates_not_duplicates(
        self, pg_conn
    ) -> None:
        """Upsert ON CONFLICT (symbol, timestamp_hour): same-hour re-run
        aktualizuje istniejący wiersz zamiast duplikować (regresja #7)."""
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_logs WHERE symbol = 'UPSERT_PL'")
            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction, timestamp) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (symbol, timestamp_hour) "
                "DO UPDATE SET price_at_prediction = EXCLUDED.price_at_prediction",
                ("UPSERT_PL", 100.0, "2026-06-03T11:05:00+00:00"),
            )
            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction, timestamp) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (symbol, timestamp_hour) "
                "DO UPDATE SET price_at_prediction = EXCLUDED.price_at_prediction",
                ("UPSERT_PL", 250.0, "2026-06-03T11:50:00+00:00"),
            )
            cur.execute(
                "SELECT count(*), max(price_at_prediction) FROM prediction_logs "
                "WHERE symbol = 'UPSERT_PL'"
            )
            row = cur.fetchone()
        assert row is not None
        # Jeden logiczny wiersz, zaktualizowana cena (250), nie duplikat.
        assert row[0] == 1
        assert float(row[1]) == pytest.approx(250.0)

    # -- Migracja 019: model_scorecards (karta kondycji modelu, #12) ----------

    def test_model_scorecards_has_symbol_column(self, pg_conn) -> None:
        """Korekta krytyczna weryfikatora: trening jest PER-SYMBOL (~43 przebiegi
        nadpisują jeden plik .ubj), więc bez kolumny `symbol` scorecardy są
        nierozróżnialne — nie wiadomo, który symbol zaakceptowano, a który
        odrzucono. Kolumna `symbol` jest wymaganiem, nie opcją."""
        cols = _columns(pg_conn, "model_scorecards")
        assert "symbol" in cols, "Brak kolumny symbol w model_scorecards (migracja 019)"

    # -- Migracja 020: decision_receipts (kwity decyzyjne, #13) ---------------

    def test_decision_receipts_column_is_nullable(self, pg_conn) -> None:
        """Kolumna JSONB `decision_receipts` MUSI być nullable — stare wiersze
        (sprzed migracji) i predykcje bez kwitów renderują się jak dziś. INSERT
        do prediction_logs bez tej kolumny musi nadal działać."""
        cols = _columns(pg_conn, "prediction_logs")
        assert "decision_receipts" in cols, (
            "Brak kolumny decision_receipts w prediction_logs (migracja 020)"
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol) VALUES (%s) RETURNING id",
                ("TEST_RECEIPTS_NULL",),
            )
            row = cur.fetchone()
        assert row is not None

    # -- Migracja 021: shock_alerts (alerty szoku poza cyklem, #11) -----------

    def test_shock_alerts_dedups_per_symbol_per_day(self, pg_conn) -> None:
        """UNIQUE(symbol, alert_date): twardy debounce "jeden alert per symbol
        per dzień". Drugi INSERT dla tej samej pary (symbol, alert_date) MUSI
        rzucić UniqueViolation — bez tego spam przy chaotycznym rynku uczy
        użytkownika ignorować kanał. Wzorzec jak
        `test_price_snapshots_unique_constraint_rejects_same_hour_duplicate`:
        autocommit=True → każdy statement to osobna transakcja, więc złapany
        wyjątek nie zatruwa kontenera współdzielonego przez inne testy."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO shock_alerts (symbol, alert_date, delta, direction) "
                "VALUES (%s, %s, %s, %s)",
                ("SHOCK_DUP", "2026-06-05", -8.0, "DOWN"),
            )
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO shock_alerts "
                    "(symbol, alert_date, delta, direction) "
                    "VALUES (%s, %s, %s, %s)",
                    ("SHOCK_DUP", "2026-06-05", -9.0, "DOWN"),
                )

    # -- Migracja 022: alpha_fusion_score (fuzja sygnałów alfa, #14) ----------

    def test_ml_feature_store_exposes_alpha_fusion_score(self, pg_conn) -> None:
        """Ósma cecha ML: `alpha_fusion_score` musi być widoczna w widoku
        (migracja 022 przebudowuje MATERIALIZED VIEW dokładając kolumnę)."""
        cols = _columns(pg_conn, "ml_feature_store")
        assert "alpha_fusion_score" in cols, (
            "Brak kolumny alpha_fusion_score w ml_feature_store (migracja 022)"
        )

    def test_alpha_fusion_score_defaults_to_zero_not_null_in_view(
        self, pg_conn
    ) -> None:
        """Ochrona przed train/serve-skew: w widoku NULL musi wchodzić jako 0.0.

        Bez `COALESCE(alpha_fusion_score, 0.0)` `dropna` przy treningu wyciąłby
        CAŁĄ historyczną część danych, gdzie kolumna jest NULL (dodana późno).
        Ten test zamraża tę ochronę: wiersz z alpha_fusion_score = NULL po
        REFRESH musi pokazać w widoku 0.0, nie NULL.
        """
        with pg_conn.cursor() as cur:
            # Izolacja: własny symbol + sprzątanie (kontener module-scoped).
            cur.execute("DELETE FROM prediction_logs WHERE symbol = 'AFS_ZERO'")
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, price_at_prediction, actual_price_after_12h, "
                " alpha_fusion_score) "
                "VALUES (%s, %s, %s, NULL) RETURNING id",
                ("AFS_ZERO", 100.0, 110.0),
            )
            row_id = cur.fetchone()[0]
            # Widok jest zmaterializowany — trzeba go odświeżyć, by zobaczyć wiersz.
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT alpha_fusion_score FROM ml_feature_store WHERE id = %s",
                (row_id,),
            )
            result = cur.fetchone()
        assert result is not None, "Wiersz nie pojawił się w widoku po REFRESH"
        assert result[0] == 0.0, (
            f"alpha_fusion_score = {result[0]!r}, oczekiwano 0.0 (COALESCE)"
        )

    # -- Migracja 023: positions (realny portfel użytkownika, #15) ------------

    def test_positions_table_exists(self, pg_conn) -> None:
        cols = _columns(pg_conn, "positions")
        required = {"id", "symbol", "quantity", "avg_cost", "as_of", "created_at"}
        assert required <= cols, f"Brakujące kolumny: {required - cols}"

    # -- Migracja 024: attestation (commit-reveal track record, #16) ----------

    def test_attestation_columns_are_nullable(self, pg_conn) -> None:
        """`commitment_hash` i `commitment_salt` MUSZĄ być nullable — stare
        wiersze i predykcje bez attestacji renderują się jak dziś. INSERT do
        prediction_logs bez tych kolumn musi nadal działać."""
        cols = _columns(pg_conn, "prediction_logs")
        assert {"commitment_hash", "commitment_salt"} <= cols, (
            "Brak kolumn attestacji commitment_hash/commitment_salt (migracja 024)"
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol) VALUES (%s) RETURNING id",
                ("TEST_ATTEST_NULL",),
            )
            row = cur.fetchone()
        assert row is not None

    # -- Migracja 025: implied_edge (edge vs rynek opcji, #18) ----------------

    def test_revealed_at_column_supports_the_reveal_sweep(self, pg_conn) -> None:
        """#16 — bez `revealed_at` sweep nie wie, co już ujawnił, a predykcje
        symboli usuniętych z configu nigdy nie zostałyby odsłonięte. Dla sceptyka
        wygląda to jak ukrywanie nietrafień."""
        cols = _columns(pg_conn, "prediction_logs")
        assert {"commitment_hash", "commitment_salt", "revealed_at"} <= cols

    def test_edge_sigma_column_exists(self, pg_conn) -> None:
        cols = _columns(pg_conn, "prediction_logs")
        assert "edge_sigma" in cols, (
            "Brak kolumny edge_sigma w prediction_logs (migracja 025)"
        )

    # -- Migracja 026: ml_feature_store bez atrap (ML-12, EXP-07) -------------

    @staticmethod
    def _settled_rows(
        cur: psycopg2.extensions.cursor,
        symbol: str,
        flags_json: list[str | None],
    ) -> list[int]:
        """Rozliczone predykcje `symbol` w odstępach dobowych (cena 100 + dzień),
        z podanymi flagami jakości jako JSON (None = kolumna NULL). Zwraca id
        w kolejności czasu."""
        cur.execute("DELETE FROM prediction_logs WHERE symbol = %s", (symbol,))
        ids: list[int] = []
        for day, flags in enumerate(flags_json, start=1):
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h, "
                " data_quality_flags) "
                "VALUES (%s, %s, %s, %s, %s::jsonb) RETURNING id",
                (
                    symbol,
                    f"2026-07-{day:02d}T10:00:00+00:00",
                    100.0 + day,
                    101.0 + day,
                    flags,
                ),
            )
            ids.append(cur.fetchone()[0])
        return ids

    def test_feature_store_excludes_llm_failed_and_av_exhausted_rows(
        self, pg_conn
    ) -> None:
        """Atrapa po awarii LLM i wyczerpane klucze AV nie trenują modelu, a
        `price_delta_no_prior_prediction`, wiersz czysty i wiersz z flagami NULL
        zostają (NULL nie może po cichu wypaść przez `NOT NULL`)."""
        with pg_conn.cursor() as cur:
            clean, llm_failed, av_exhausted, no_prior, null_flags = self._settled_rows(
                cur,
                "QF_EXCL",
                [
                    "[]",
                    '["llm_analysis_failed"]',
                    '["news_volume_24h_missing", "av_keys_exhausted"]',
                    '["price_delta_no_prior_prediction"]',
                    None,
                ],
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT id FROM ml_feature_store WHERE symbol = 'QF_EXCL'"
            )
            in_view = {row[0] for row in cur.fetchall()}

        assert in_view == {clean, no_prior, null_flags}
        assert not in_view & {llm_failed, av_exhausted}

    def test_feature_store_lag_runs_over_the_full_sequence(self, pg_conn) -> None:
        """Filtr jest PO LAG: wiersz po atrapie liczy `price_prev_12h` od ceny
        atrapy — tak jak inference (`get_last_prediction_price`), która flag
        nie czyta. Filtr w CTE dałby train/serve skew."""
        with pg_conn.cursor() as cur:
            _, _, after = self._settled_rows(
                cur, "QF_LAG", ["[]", '["llm_analysis_failed"]', "[]"]
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT price_prev_12h, price_prev_24h FROM ml_feature_store "
                "WHERE id = %s",
                (after,),
            )
            row = cur.fetchone()

        assert row is not None
        # Ceny: dzień 1 → 101, dzień 2 (atrapa) → 102, dzień 3 → 103.
        assert float(row[0]) == pytest.approx(102.0)
        assert float(row[1]) == pytest.approx(101.0)

    def test_feature_store_rebuild_keeps_indexes_for_concurrent_refresh(
        self, pg_conn
    ) -> None:
        """DROP MATERIALIZED VIEW kasuje indeksy; bez unikalnego po `id`
        `refresh_ml_feature_store` (REFRESH ... CONCURRENTLY) pada w Slow Loopie."""
        assert {
            "idx_ml_feature_store_id",
            "idx_ml_feature_store_symbol_timestamp",
        } <= _indexes(pg_conn, "ml_feature_store")
        with pg_conn.cursor() as cur:
            cur.execute("SELECT refresh_ml_feature_store()")

    # -- Migracja 027: feature_set_version (wersja zestawu cech, DAT-09) ------

    def test_feature_set_version_reaches_the_feature_store(self, pg_conn) -> None:
        """Trening filtruje po `feature_set_version` z widoku — kolumna musi być
        w obu miejscach, a NULL („wersja nieznana") nie może udawać wersji 1."""
        assert "feature_set_version" in _columns(pg_conn, "prediction_logs")
        assert "feature_set_version" in _columns(pg_conn, "ml_feature_store")
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_logs WHERE symbol LIKE 'FSV_%'")
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h, "
                " feature_set_version) VALUES "
                "('FSV_V2', '2026-11-02T10:00:00+00:00', 100.0, 101.0, 2), "
                "('FSV_NULL', '2026-11-02T10:00:00+00:00', 100.0, 101.0, NULL)"
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT symbol, feature_set_version FROM ml_feature_store "
                "WHERE symbol LIKE 'FSV_%'"
            )
            versions = dict(cur.fetchall())

        assert versions == {"FSV_V2": 2, "FSV_NULL": None}

    # -- Migracja 028: baseline'y dryfu i „sam LLM" w karcie modelu (ML-10) ---

    def test_model_scorecards_store_the_drift_and_llm_baselines(self, pg_conn) -> None:
        """Karta pokazuje wszystkie trzy baseline'y bramki; nowe kolumny są
        nullable, bo przebieg bez metryk (skipped) i stare wiersze ich nie mają."""
        assert {"drift_baseline_rmse", "llm_baseline_rmse"} <= _columns(
            pg_conn, "model_scorecards"
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status) VALUES (%s, %s) "
                "RETURNING drift_baseline_rmse, llm_baseline_rmse",
                ("ALL", "skipped"),
            )
            row = cur.fetchone()
        assert row == (None, None)

    # -- Migracja 029: moment rozliczenia predykcji (WIA-19) ------------------

    def test_settling_a_prediction_stamps_settled_at(self, pg_conn) -> None:
        """Okna „rozliczone w oknie” potrzebują chwili rozliczenia, nie utworzenia:
        piątkowa predykcja rozliczona w poniedziałek ma wpaść do poniedziałku."""
        assert "settled_at" in _columns(pg_conn, "prediction_logs")
        assert "idx_prediction_logs_settled_at" in _indexes(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('SET_STAMP', '2026-10-16T15:00:00+00:00', 100.0) "
                "RETURNING settled_at"
            )
            assert cur.fetchone() == (None,)

            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 101.0, "
                "is_trend_correct = true WHERE symbol = 'SET_STAMP' "
                "RETURNING settled_at, settled_at >= now() - INTERVAL '1 minute'"
            )
            settled_at, is_fresh = cur.fetchone()

        assert settled_at is not None
        assert is_fresh is True

    def test_resettling_does_not_restamp_settled_at(self, pg_conn) -> None:
        """Druga ocena już rozliczonego wiersza nie przesuwa chwili rozliczenia —
        inaczej predykcja wracałaby do okna „rozliczone w tym cyklu” co cykl."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('SET_RESTAMP', '2026-10-16T15:00:00+00:00', 100.0)"
            )
            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 101.0 "
                "WHERE symbol = 'SET_RESTAMP'"
            )
            # Znany stempel (osobne UPDATE: trigger go nie nadpisuje), żeby
            # porównanie nie zależało od rozdzielczości zegara.
            cur.execute(
                "UPDATE prediction_logs SET settled_at = '2026-10-19T05:30:00+00:00' "
                "WHERE symbol = 'SET_RESTAMP'"
            )
            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 102.0, "
                "correction_insights = 'druga ocena' WHERE symbol = 'SET_RESTAMP' "
                "RETURNING settled_at = '2026-10-19T05:30:00+00:00'::timestamptz"
            )
            unchanged = cur.fetchone()[0]

        assert unchanged is True

    # -- Migracja 030: wersja przepisu predykcji (WIA-20) ---------------------

    def test_recipe_version_column_is_nullable_and_indexed(self, pg_conn) -> None:
        """Track record segmentuje się po przepisie; stare wiersze i zapis bez
        flagi `recipe_version_enabled` zostają NULL („przepis nieznany”)."""
        assert "recipe_version" in _columns(pg_conn, "prediction_logs")
        assert "idx_prediction_logs_recipe_version" in _indexes(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('RCP_NULL', '2026-10-19T05:30:00+00:00', 100.0) "
                "RETURNING recipe_version"
            )
            assert cur.fetchone() == (None,)
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, recipe_version) "
                "VALUES ('RCP_SET', '2026-10-19T05:30:00+00:00', 100.0, %s) "
                "RETURNING recipe_version",
                ("a" * 64,),
            )
            assert cur.fetchone() == ("a" * 64,)

    # -- Migracja 031: wektor cech point-in-time w każdym cyklu (ML-14) -------

    def test_feature_snapshots_unique_per_symbol_hour(self, pg_conn) -> None:
        """Re-run cyklu w tej samej godzinie aktualizuje wektor (upsert na
        `symbol,as_of_hour`), a goły INSERT drugiego wiersza tej godziny rzuca."""
        assert {
            "id", "symbol", "as_of", "as_of_hour",
            "feature_set_version", "features", "gated",
        } <= _columns(pg_conn, "feature_snapshots")
        assert {
            "idx_feature_snapshots_symbol_hour_uniq",
            "idx_feature_snapshots_as_of",
            "idx_feature_snapshots_version_gated",
        } <= _indexes(pg_conn, "feature_snapshots")
        upsert = (
            "INSERT INTO feature_snapshots "
            "(symbol, as_of, feature_set_version, features, gated) "
            "VALUES (%s, %s, 2, %s::jsonb, %s) "
            "ON CONFLICT (symbol, as_of_hour) "
            "DO UPDATE SET features = EXCLUDED.features, gated = EXCLUDED.gated"
        )
        with pg_conn.cursor() as cur:
            cur.execute(upsert, ("FS_HOUR", "2026-10-19T05:10:00+00:00", '{"price": 1}', False))
            cur.execute(upsert, ("FS_HOUR", "2026-10-19T05:50:00+00:00", '{"price": 2}', True))
            cur.execute(
                "SELECT count(*), max((features->>'price')::int), bool_and(gated), "
                "max(as_of_hour) FROM feature_snapshots WHERE symbol = 'FS_HOUR'"
            )
            count, price, gated, hour = cur.fetchone()
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO feature_snapshots "
                    "(symbol, as_of, feature_set_version, features, gated) "
                    "VALUES ('FS_HOUR', '2026-10-19T05:59:00+00:00', 2, '{}'::jsonb, false)"
                )

        assert (count, price, gated) == (1, 2, True)
        assert hour == datetime(2026, 10, 19, 5, 0, tzinfo=UTC)

    def test_feature_snapshots_require_version_vector_and_gate(self, pg_conn) -> None:
        """Wiersz bez wersji zestawu cech, wektora albo decyzji bramki jest
        bezużyteczny dla treningu i raportu selekcji — baza go odrzuca."""
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.NotNullViolation):
            cur.execute(
                "INSERT INTO feature_snapshots (symbol, features, gated) "
                "VALUES ('FS_NULL', '{}'::jsonb, false)"
            )

    def test_training_reads_back_the_saved_feature_vector(self, pg_conn) -> None:
        """ML-52: trening bierze wektor z JSONB bez przeliczania — po zapisie (JSON
        jak z PostgREST) i odczycie kolumnami adaptera wektor modelu jest IDENTYCZNY
        z tym, który dostał serving, a etykieta to zwrot do pierwszego snapshotu
        ceny ≥ 12 h później."""
        import json

        from src.application.ml_features import llm_model_vector
        from src.domain.training_labels import FeatureSnapshot, label_snapshots

        served = {
            "price_delta": 0.1 + 0.2,
            "av_sentiment_score": -0.123456789012345,
            "av_relevance_avg": 1e-7,
            "news_volume_24h": 12.0,
            "high_relevance_count": 3.0,
            "llm_trend_signal": -1.0,
            "av_llm_agreement": 0.5,
        }
        vector = {
            **served,
            "price": 187.1234,
            "alpha_fusion_score": None,
            "regime": "RISK_ON",
            "data_quality_flags": ["price_delta_no_prior_prediction"],
        }
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feature_snapshots "
                "(symbol, as_of, feature_set_version, features, gated) "
                "VALUES ('FS_ML52', '2026-10-20T07:30:00+00:00', 4, %s::jsonb, true)",
                (json.dumps(vector),),
            )
            for stamp, price in (
                ("2026-10-20T07:29:59+00:00", 187.1234),
                ("2026-10-20T15:00:00+00:00", 150.0),
                ("2026-10-21T07:30:00+00:00", 190.0),
            ):
                cur.execute(
                    "INSERT INTO price_snapshots (symbol, timestamp, price) "
                    "VALUES ('FS_ML52', %s, %s)",
                    (stamp, price),
                )
            cur.execute(
                "SELECT symbol, as_of, gated, features, feature_set_version "
                "FROM feature_snapshots WHERE symbol = 'FS_ML52' AND feature_set_version = 4 "
                "ORDER BY as_of, id"
            )
            [(symbol, as_of, gated, features, version)] = cur.fetchall()
            cur.execute(
                "SELECT timestamp, price FROM price_snapshots "
                "WHERE symbol = 'FS_ML52' ORDER BY timestamp, id"
            )
            series = [(stamp, float(price)) for stamp, price in cur.fetchall()]

        assert features == vector
        snapshot = FeatureSnapshot(symbol, as_of, gated, features, version)
        assert llm_model_vector(snapshot.features) == served
        [labeled] = label_snapshots([snapshot], {"FS_ML52": series})
        assert labeled.target_return == pytest.approx(190.0 / 187.1234 - 1)

    # -- Migracja 032: przedział ruchu z conformal prediction (ML-34) ---------

    def test_move_interval_columns_are_nullable_prices(self, pg_conn) -> None:
        """Stare wiersze i zapis bez flagi `interval_coverage_enabled` zostają NULL;
        przedział ląduje w tej samej skali co `predicted_target_price`."""
        assert {"interval_low", "interval_high"} <= _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('MI_NULL', '2026-10-19T05:30:00+00:00', 100.0) "
                "RETURNING interval_low, interval_high"
            )
            assert cur.fetchone() == (None, None)
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, interval_low, interval_high) "
                "VALUES ('MI_SET', '2026-10-19T05:30:00+00:00', 100.0, 97.91234, 102.4) "
                "RETURNING interval_low, interval_high"
            )
            low, high = cur.fetchone()

        assert (float(low), float(high)) == (97.9123, 102.4)

    # -- Migracja 033: stan histerezy reżimu (ALR-03) -------------------------

    def test_regime_history_reads_newest_cycle_first(self, pg_conn) -> None:
        """Fast Loop czyta najnowszy wiersz po `cycle_at` (indeks malejący);
        reżimy spoza `MarketRegime` i ujemna seria są odrzucane."""
        assert {"id", "cycle_at", "raw_regime", "regime", "streak"} <= _columns(
            pg_conn, "regime_history"
        )
        assert "idx_regime_history_cycle_at" in _indexes(pg_conn, "regime_history")
        insert = (
            "INSERT INTO regime_history (cycle_at, raw_regime, regime, streak) "
            "VALUES (%s, %s, %s, %s)"
        )
        with pg_conn.cursor() as cur:
            cur.execute(insert, ("2026-10-19T05:30:00+00:00", "RISK_ON", "RISK_OFF", 1))
            cur.execute(insert, ("2026-10-19T13:30:00+00:00", "RISK_ON", "RISK_ON", 0))
            cur.execute(
                "SELECT raw_regime, regime, streak FROM regime_history "
                "ORDER BY cycle_at DESC, id DESC LIMIT 1"
            )
            assert cur.fetchone() == ("RISK_ON", "RISK_ON", 0)
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(insert, ("2026-10-19T14:30:00+00:00", "PANIC", "RISK_OFF", 0))
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(insert, ("2026-10-19T14:30:00+00:00", "RISK_ON", "RISK_OFF", -1))

    # -- Migracja 034: delta bramki volatility (EXP-01) ------------------------

    def test_gate_delta_column_is_a_nullable_fraction(self, pg_conn) -> None:
        """Zapis bez flagi `gate_delta_enabled` zostawia NULL; delta to ułamek."""
        assert "gate_delta" in _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('GD_NULL', '2026-10-19T05:30:00+00:00', 100.0) "
                "RETURNING gate_delta"
            )
            assert cur.fetchone() == (None,)
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction, gate_delta) "
                "VALUES ('GD_SET', '2026-10-19T05:30:00+00:00', 100.0, -0.0312345) "
                "RETURNING gate_delta"
            )
            assert cur.fetchone()[0] == pytest.approx(-0.0312345)

    # -- Migracja 035: waluta i data zakupu pozycji (PORT-18) ------------------

    def test_positions_have_an_iso_currency_and_an_optional_purchase_date(
        self, pg_conn
    ) -> None:
        """Wiersz bez waluty to USD, bez daty zakupu — NULL; waluta spoza ISO odpada."""
        assert {"currency", "purchase_date"} <= _columns(pg_conn, "positions")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost) "
                "VALUES ('CUR_DEFAULT', 1, 10) RETURNING currency, purchase_date"
            )
            assert cur.fetchone() == ("USD", None)
            cur.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost, currency, purchase_date) "
                "VALUES ('CUR_SET', 1, 10, 'EUR', '2025-03-01') "
                "RETURNING currency, purchase_date::TEXT"
            )
            assert cur.fetchone() == ("EUR", "2025-03-01")
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO positions (symbol, quantity, avg_cost, currency) "
                    "VALUES ('CUR_BAD', 1, 10, 'usd')"
                )

    # -- Migracja 036: cache kursów NBP z dnia (PORT-11) -----------------------

    def test_fx_rates_cache_one_rate_per_currency_and_day(self, pg_conn) -> None:
        """Klucz (currency, on_date); kurs dodatni, z tabeli nie późniejszej niż dzień."""
        assert {"currency", "on_date", "effective_date", "rate", "source", "fetched_at"} <= (
            _columns(pg_conn, "fx_rates")
        )
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fx_rates (currency, on_date, effective_date, rate) "
                "VALUES ('USD', '2025-03-02', '2025-02-28', 3.9725) RETURNING source"
            )
            assert cur.fetchone() == ("NBP_A",)
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cur.execute(
                    "INSERT INTO fx_rates (currency, on_date, effective_date, rate) "
                    "VALUES ('USD', '2025-03-02', '2025-02-28', 3.99)"
                )
            for values in (
                "('usd', '2025-03-03', '2025-03-03', 3.9)",
                "('EUR', '2025-03-03', '2025-03-03', 0)",
                "('EUR', '2025-03-03', '2025-03-04', 4.1)",
            ):
                with pytest.raises(psycopg2.errors.CheckViolation):
                    cur.execute(
                        "INSERT INTO fx_rates (currency, on_date, effective_date, rate) "
                        f"VALUES {values}"
                    )

    # -- Migracja 037: ramiona eksperymentów (EXP-05) --------------------------

    def test_experiment_arms_column_is_a_nullable_json_object(self, pg_conn) -> None:
        """Zapis bez flagi `experiment_arms_enabled` zostawia NULL; ramiona to obiekt JSON."""
        assert "experiment_arms" in _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('EA_NULL', '2026-11-02T21:30:00+00:00', 100.0) "
                "RETURNING experiment_arms"
            )
            assert cur.fetchone() == (None,)
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, experiment_arms) "
                "VALUES ('EA_SET', '2026-11-02T21:30:00+00:00', 100.0, "
                "'{\"EXP-02\": \"blind\"}') "
                "RETURNING experiment_arms ->> 'EXP-02'"
            )
            assert cur.fetchone() == ("blind",)

    # -- Migracja 039: ml_feature_store bez niepełnego feedu AV (DAT-07, DAT-08) -

    def test_feature_store_excludes_av_degraded_rows(self, pg_conn) -> None:
        """Payload bez feedu i błąd sieci w środku fetchu dają zera „bez danych",
        tak jak wyczerpane klucze — nie trenują modelu. Filtry 026 zostają."""
        with pg_conn.cursor() as cur:
            clean, unexpected, fetch_error, exhausted, no_prior = self._settled_rows(
                cur,
                "QF_AV039",
                [
                    "[]",
                    '["av_unexpected_payload"]',
                    '["news_volume_24h_missing", "av_fetch_error"]',
                    '["av_keys_exhausted"]',
                    '["price_delta_no_prior_prediction"]',
                ],
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute("SELECT id FROM ml_feature_store WHERE symbol = 'QF_AV039'")
            in_view = {row[0] for row in cur.fetchall()}

        assert in_view == {clean, no_prior}
        assert not in_view & {unexpected, fetch_error, exhausted}

    def test_feature_store_lag_runs_over_av_degraded_rows(self, pg_conn) -> None:
        """Filtr 039 też jest PO LAG: wiersz po `av_fetch_error` liczy
        `price_prev_12h` od ceny tego wiersza, jak inference."""
        with pg_conn.cursor() as cur:
            _, _, after = self._settled_rows(
                cur, "QF_AV039_LAG", ["[]", '["av_fetch_error"]', "[]"]
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT price_prev_12h, price_prev_24h FROM ml_feature_store WHERE id = %s",
                (after,),
            )
            row = cur.fetchone()

        assert row is not None
        assert float(row[0]) == pytest.approx(102.0)
        assert float(row[1]) == pytest.approx(101.0)

    def test_feature_store_after_039_keeps_the_feature_contract(self, pg_conn) -> None:
        """039 odtwarza widok 1:1 z 027 — te same kolumny i oba indeksy."""
        assert {
            "price_delta",
            "av_sentiment_score",
            "news_volume_24h",
            "alpha_fusion_score",
            "feature_set_version",
            "target_return",
        } <= _columns(pg_conn, "ml_feature_store")
        assert {
            "idx_ml_feature_store_id",
            "idx_ml_feature_store_symbol_timestamp",
        } <= _indexes(pg_conn, "ml_feature_store")

    # -- Migracja 040: ledger kosztów LLM per wywołanie (FIN-13) ---------------

    def test_llm_calls_holds_tokens_and_cost_but_no_prompt(self, pg_conn) -> None:
        """Wiersz to liczby i identyfikatory przebiegu; żadnej kolumny na treść
        promptu ani odpowiedzi (kontekst portfela)."""
        assert _columns(pg_conn, "llm_calls") == {
            "id",
            "run_id",
            "symbol",
            "node",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
            "price_as_of",
            "created_at",
        }

    def test_llm_calls_is_indexed_by_created_at(self, pg_conn) -> None:
        assert "idx_llm_calls_created_at" in _indexes(pg_conn, "llm_calls")

    def test_llm_calls_accepts_a_call_without_symbol_cost_or_price(self, pg_conn) -> None:
        """Slow Loop nie ma symbolu, a model spoza cennika nie ma kosztu."""
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO llm_calls (run_id, node, provider, model, input_tokens, "
                "output_tokens) VALUES ('LC_RUN', 'weekly_recap', 'openai', 'gpt-9', 10, 5) "
                "RETURNING symbol, cost_usd, price_as_of, cache_read_tokens, "
                "cache_write_tokens, created_at IS NOT NULL"
            )
            assert cur.fetchone() == (None, None, None, 0, 0, True)

    def test_llm_calls_rejects_negative_token_counts(self, pg_conn) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO llm_calls (run_id, node, provider, model, input_tokens, "
                "output_tokens) VALUES ('LC_NEG', 'predict', 'anthropic', 'c', -1, 0)"
            )

    # -- Migracja 041: historia przebiegów pętli (NIEZ-29) ---------------------

    def test_cycle_runs_holds_the_run_identity_and_counts(self, pg_conn) -> None:
        """Jeden wiersz na przebieg: tożsamość runu (łączy się z `llm_calls` po
        `run_id`), liczby cyklu i stan maila — bez symboli i treści raportu."""
        assert _columns(pg_conn, "cycle_runs") == {
            "id",
            "loop",
            "run_id",
            "run_attempt",
            "started_at",
            "duration_s",
            "eligible",
            "saved",
            "failures",
            "paid_calls",
            "report_delivered",
            "created_at",
        }

    def test_cycle_runs_is_indexed_by_started_at(self, pg_conn) -> None:
        assert "idx_cycle_runs_started_at" in _indexes(pg_conn, "cycle_runs")

    def test_cycle_runs_accepts_a_local_run_without_attempt_or_email(self, pg_conn) -> None:
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cycle_runs (loop, run_id, started_at, duration_s, eligible, "
                "saved, failures) VALUES ('fast', 'local', NOW(), 12.5, 3, 1, 0) "
                "RETURNING run_attempt, report_delivered, paid_calls, created_at IS NOT NULL"
            )
            assert cur.fetchone() == (None, None, {}, True)

    def test_cycle_runs_rejects_an_unknown_loop_and_negative_counts(self, pg_conn) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO cycle_runs (loop, run_id, started_at, duration_s, eligible, "
                "saved, failures) VALUES ('nightly', 'CR_BAD', NOW(), 1, 1, 0, 0)"
            )
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO cycle_runs (loop, run_id, started_at, duration_s, eligible, "
                "saved, failures) VALUES ('fast', 'CR_NEG', NOW(), 1, 1, -1, 0)"
            )

    # -- Migracja 042: sygnał referencyjny WIA-24 ------------------------------

    def test_reference_predictions_hold_read_call_and_settlement_in_one_row(
        self, pg_conn
    ) -> None:
        """Osobna tabela, nie `prediction_logs`: żaden odczyt ścieżek LLM (reflect,
        RAG, widok cech, track record) jej nie widzi."""
        assert _columns(pg_conn, "reference_predictions") == {
            "id",
            "symbol",
            "read_at",
            "price",
            "predicted_trend",
            "recipe_version",
            "outcome",
            "settled_price",
            "settled_at",
            "created_at",
        }

    def test_reference_predictions_are_indexed_for_the_weekly_reads(self, pg_conn) -> None:
        assert {
            "idx_reference_predictions_read_at",
            "idx_reference_predictions_open_calls",
        } <= _indexes(pg_conn, "reference_predictions")

    def test_reference_read_defaults_to_reference_v1_once_per_symbol_and_time(
        self, pg_conn
    ) -> None:
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reference_predictions (symbol, read_at, price) "
                "VALUES ('REF_A', '2026-10-25T03:00:00Z', 100.5) "
                "RETURNING recipe_version, predicted_trend, outcome, created_at IS NOT NULL"
            )
            assert cur.fetchone() == ("reference-v1", None, None, True)
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(
                "INSERT INTO reference_predictions (symbol, read_at, price) "
                "VALUES ('REF_A', '2026-10-25T03:00:00Z', 101)"
            )

    def test_reference_settlement_round_trips(self, pg_conn) -> None:
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reference_predictions (symbol, read_at, price, predicted_trend) "
                "VALUES ('REF_B', '2026-10-18T03:00:00Z', 100, 'BULLISH') RETURNING id"
            )
            (row_id,) = cur.fetchone()
            cur.execute(
                "UPDATE reference_predictions SET outcome = 'HIT', settled_price = 104, "
                "settled_at = '2026-10-25T03:00:00Z' WHERE id = %s RETURNING outcome",
                (row_id,),
            )
            assert cur.fetchone() == ("HIT",)

    @pytest.mark.parametrize(
        "values",
        [
            "('REF_C1', NOW(), 0, NULL, NULL, NULL, NULL, 'reference-v1')",
            "('REF_C2', NOW(), 100, 'SIDEWAYS', NULL, NULL, NULL, 'reference-v1')",
            "('REF_C3', NOW(), 100, NULL, 'HIT', 104, NOW(), 'reference-v1')",
            "('REF_C4', NOW(), 100, 'BULLISH', 'HIT', NULL, NOW(), 'reference-v1')",
            "('REF_C5', NOW(), 100, 'BULLISH', 'EXPIRED', 104, NOW(), 'reference-v1')",
            "('REF_C6', NOW(), 100, 'BULLISH', NULL, 104, NULL, 'reference-v1')",
            "('REF_C7', NOW(), 100, 'BULLISH', 'HIT', 104, NULL, 'reference-v1')",
            "('REF_C8', NOW(), 100, 'BULLISH', NULL, NULL, NULL, 'llm-abc')",
        ],
        ids=[
            "price_not_positive",
            "sideways_call",
            "outcome_without_call",
            "hit_without_price",
            "expired_with_price",
            "price_without_outcome",
            "outcome_without_time",
            "not_a_reference_label",
        ],
    )
    def test_reference_predictions_reject_inconsistent_rows(self, pg_conn, values) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO reference_predictions (symbol, read_at, price, predicted_trend, "
                f"outcome, settled_price, settled_at, recipe_version) VALUES {values}"
            )

    # -- Migracja 043: dosłowny plaintext commitmentu (WIA-70) ------------------

    def test_commitment_payload_column_is_nullable_text(self, pg_conn) -> None:
        """Reveal publikuje zapisany plaintext bajt w bajt: TEXT, nie JSONB (JSONB
        przestawia klucze i białe znaki). NULL dla wierszy sprzed 043 i predykcji
        bez atestacji — tych sweep ujawnia z kolumn, jak dotąd."""
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT data_type, is_nullable FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'prediction_logs'
                  AND column_name = 'commitment_payload'
                """
            )
            assert cur.fetchone() == ("text", "YES")

    def test_commitment_payload_keeps_the_plaintext_byte_for_byte(self, pg_conn) -> None:
        plaintext = '{"price_at_prediction":"90.1112","schema_version":2,"symbol":"Zażółć"}'
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, commitment_payload) VALUES (%s, %s) "
                "RETURNING commitment_payload",
                ("TEST_ATTEST_PAYLOAD", plaintext),
            )
            assert cur.fetchone() == (plaintext,)

    # -- Migracja 044: wersja reguły confidence_calibration (WIA-37) -----------

    def test_calibration_score_version_backfills_scored_rows_as_linear(
        self, pg_conn
    ) -> None:
        """Ocena bez wersji to reguła liniowa (v1); wiersz bez oceny zostaje NULL,
        a wiersz z v2 nie wraca do v1 przy ponownym przebiegu migracji."""
        assert "calibration_score_version" in _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, confidence_calibration) VALUES "
                "('CSV_V1', '2026-11-03T21:30:00+00:00', 100.0, 0.4), "
                "('CSV_NONE', '2026-11-03T21:30:00+00:00', 100.0, NULL)"
            )
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction, "
                "confidence_calibration, calibration_score_version) "
                "VALUES ('CSV_V2', '2026-11-03T21:30:00+00:00', 100.0, 0.19, 2)"
            )
            cur.execute(
                (MIGRATIONS_DIR / "044_calibration_score_version.sql").read_text()
            )
            cur.execute(
                "SELECT symbol, calibration_score_version FROM prediction_logs "
                "WHERE symbol LIKE 'CSV_%' ORDER BY symbol"
            )
            assert cur.fetchall() == [("CSV_NONE", None), ("CSV_V1", 1), ("CSV_V2", 2)]

    def test_calibration_score_version_rejects_an_unknown_rule(self, pg_conn) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, calibration_score_version) "
                "VALUES ('CSV_BAD', '2026-11-03T21:30:00+00:00', 100.0, 3)"
            )

    # -- Migracja 047: LAG po wszystkich zalogowanych predykcjach (ML-17) -------

    def test_price_delta_lags_over_all_logged_predictions_not_only_settled(
        self, pg_conn
    ) -> None:
        """Inference bierze cenę ostatniej ZALOGOWANEJ predykcji, rozliczonej czy
        nie. Ceny 100 (rozliczona), 110 (otwarta), 121 (rozliczona): delta trzeciej
        to 121/110 − 1 = 0,10, a LAG po samych rozliczonych dawał 121/100 − 1 = 0,21."""
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_logs WHERE symbol = 'QF_LAG047'")
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h) VALUES "
                "('QF_LAG047', '2026-08-01T10:00:00+00:00', 100.0, 101.0), "
                "('QF_LAG047', '2026-08-02T10:00:00+00:00', 110.0, NULL), "
                "('QF_LAG047', '2026-08-03T10:00:00+00:00', 121.0, 122.0)"
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT price_current, price_prev_12h, price_prev_24h, price_delta "
                "FROM ml_feature_store WHERE symbol = 'QF_LAG047' ORDER BY timestamp"
            )
            rows = [tuple(float(v) if v is not None else None for v in r) for r in cur]

        # Otwarta predykcja nie ma etykiety, więc nie trenuje — ale jest w LAG.
        assert [row[0] for row in rows] == [100.0, 121.0]
        assert rows[1][1] == pytest.approx(110.0)
        assert rows[1][2] == pytest.approx(100.0)
        assert rows[1][3] == pytest.approx(0.10)

    def test_feature_store_after_047_keeps_the_feature_contract(self, pg_conn) -> None:
        """047 odtwarza widok 1:1 z 039 (kolumny, oba indeksy, REFRESH CONCURRENTLY)
        i opisuje prawdziwą semantykę `price_prev_12h` / `price_prev_24h`."""
        assert {
            "price_delta",
            "av_sentiment_score",
            "news_volume_24h",
            "alpha_fusion_score",
            "feature_set_version",
            "target_return",
        } <= _columns(pg_conn, "ml_feature_store")
        assert {
            "idx_ml_feature_store_id",
            "idx_ml_feature_store_symbol_timestamp",
        } <= _indexes(pg_conn, "ml_feature_store")
        with pg_conn.cursor() as cur:
            cur.execute("SELECT refresh_ml_feature_store()")
            cur.execute(
                "SELECT col_description('ml_feature_store'::regclass, attnum) "
                "FROM pg_attribute WHERE attrelid = 'ml_feature_store'::regclass "
                "AND attname = 'price_prev_12h'"
            )
            comment = cur.fetchone()[0]

        assert "POPRZEDNIEJ zalogowanej predykcji" in comment

    # -- Migracja 048: kanoniczny predicted_trend + CHECK NOT VALID (ML-19) ---

    @pytest.mark.parametrize("trend", ["UP", " bullish ", "Bearish"])
    def test_predicted_trend_check_rejects_unknown_value(self, pg_conn, trend) -> None:
        """Schemat sam blokuje kolejne zatrucia: tylko kanoniczne wartości."""
        with _rolled_back(pg_conn) as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO prediction_logs (symbol, predicted_trend) VALUES ('TR_048', %s)",
                (trend,),
            )

    def test_predicted_trend_check_accepts_canonical_values_and_null(self, pg_conn) -> None:
        with _rolled_back(pg_conn) as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, predicted_trend) VALUES "
                "('TR_048', '2026-08-01T10:00:00+00:00', 'BULLISH'), "
                "('TR_048', '2026-08-02T10:00:00+00:00', 'BEARISH'), "
                "('TR_048', '2026-08-03T10:00:00+00:00', 'SIDEWAYS'), "
                "('TR_048', '2026-08-04T10:00:00+00:00', NULL)"
            )
            cur.execute("SELECT count(*) FROM prediction_logs WHERE symbol = 'TR_048'")
            assert cur.fetchone() == (4,)

    # -- Migracja 049: run_id przebiegu w karcie modelu (ML-24) ----------------

    def test_model_scorecards_run_id_is_a_nullable_indexed_uuid(self, pg_conn) -> None:
        """Wiersze jednego przebiegu dzieli `run_id`; wiersz bez niego (historia,
        flaga off) dalej się zapisuje."""
        assert "run_id" in _columns(pg_conn, "model_scorecards")
        assert "idx_model_scorecards_run_id" in _indexes(pg_conn, "model_scorecards")
        run_id = "5f0c6c1e-8a43-4d3a-9b0c-2f1d3e4a5b6c"
        with _rolled_back(pg_conn) as cur:
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, run_id) VALUES "
                "('ALL', 'skipped', %s), ('AAPL', 'skipped', %s), ('MSFT', 'skipped', NULL)",
                (run_id, run_id),
            )
            cur.execute(
                "SELECT symbol FROM model_scorecards WHERE run_id = %s ORDER BY symbol",
                (run_id,),
            )
            same_run = [row[0] for row in cur.fetchall()]
            cur.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'model_scorecards' AND column_name = 'run_id'"
            )
            data_type = cur.fetchone()

        assert same_run == ["AAPL", "ALL"]
        assert data_type == ("uuid",)

    # -- Migracja 050: settled_at i horyzont etykiety w widoku (ML-25) ---------

    def test_ml_feature_store_exposes_settled_at_and_label_horizon(self, pg_conn) -> None:
        """Wiersz rozliczony po 26 h ma `label_horizon_hours = 26`; dolna granica
        backfillu 029 (`settled_at = timestamp`) to brak pomiaru, nie 0 h."""
        assert {"settled_at", "label_horizon_hours"} <= _columns(pg_conn, "ml_feature_store")
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_logs WHERE symbol LIKE 'LH050_%'")
            # INSERT z jawnym settled_at — trigger 029 stempluje tylko UPDATE.
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h, settled_at) "
                "VALUES "
                "('LH050_LATE', '2026-08-07T10:00:00+00:00', 100.0, 101.0, "
                " '2026-08-08T12:00:00+00:00'), "
                "('LH050_FLOOR', '2026-08-07T10:00:00+00:00', 100.0, 101.0, "
                " '2026-08-07T10:00:00+00:00')"
            )
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT symbol, settled_at IS NOT NULL, label_horizon_hours "
                "FROM ml_feature_store WHERE symbol LIKE 'LH050_%'"
            )
            rows = {symbol: (settled, horizon) for symbol, settled, horizon in cur.fetchall()}

        assert rows["LH050_LATE"] == (True, pytest.approx(26.0))
        assert rows["LH050_FLOOR"] == (True, None)

    def test_feature_store_after_050_keeps_the_lag_and_the_indexes(self, pg_conn) -> None:
        """050 odtwarza widok z 047: LAG po wszystkich predykcjach, oba indeksy,
        komentarze kolumn i zamknięte role API (sprawdza je test emulacji ról)."""
        assert {
            "idx_ml_feature_store_id",
            "idx_ml_feature_store_symbol_timestamp",
        } <= _indexes(pg_conn, "ml_feature_store")
        with pg_conn.cursor() as cur:
            cur.execute("SELECT refresh_ml_feature_store()")
            cur.execute(
                "SELECT col_description('ml_feature_store'::regclass, attnum) "
                "FROM pg_attribute WHERE attrelid = 'ml_feature_store'::regclass "
                "AND attname = 'label_horizon_hours'"
            )
            comment = cur.fetchone()[0]

        assert "horyzont etykiety" in comment

    # -- Migracja 051: rozkład trendu predykcji (ML-29) ------------------------

    def test_prediction_logs_trend_probabilities_is_a_nullable_jsonb(self, pg_conn) -> None:
        """Predykcja z rozkładem i bez niego (flaga off, fallback) zapisują się
        obok siebie; p klasy czyta się zapytaniem JSONB."""
        assert "trend_probabilities" in _columns(pg_conn, "prediction_logs")
        distribution = '{"BULLISH": 0.2, "BEARISH": 0.1, "SIDEWAYS": 0.7}'
        with _rolled_back(pg_conn) as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, predicted_trend, "
                "trend_probabilities) VALUES "
                "('TP_051', '2026-10-01T10:00:00+00:00', 'SIDEWAYS', %s::jsonb), "
                "('TP_051', '2026-10-02T10:00:00+00:00', 'BULLISH', NULL)",
                (distribution,),
            )
            cur.execute(
                "SELECT (trend_probabilities ->> 'SIDEWAYS')::float FROM prediction_logs "
                "WHERE symbol = 'TP_051' AND trend_probabilities IS NOT NULL"
            )
            sideways = cur.fetchone()
            cur.execute("SELECT count(*) FROM prediction_logs WHERE symbol = 'TP_051'")
            count = cur.fetchone()

        assert sideways == (pytest.approx(0.7),)
        assert count == (2,)

    # -- Migracja 052: rozkład cech treningu w karcie modelu (ML-21) -----------

    def test_model_scorecards_feature_stats_is_a_nullable_jsonb(self, pg_conn) -> None:
        """Wiersz z `feature_stats` (biny + PSI) i wiersz bez niego (historia,
        flaga off) zapisują się obok siebie; PSI czyta się zapytaniem JSONB."""
        assert "feature_stats" in _columns(pg_conn, "model_scorecards")
        stats = '{"price_delta": {"edges": [0.0], "fractions": [0.5, 0.5], "psi_recent_30d": 0.31}}'
        with _rolled_back(pg_conn) as cur:
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, feature_stats) VALUES "
                "('ALL', 'trained_successfully', %s::jsonb), ('AAPL', 'skipped', NULL)",
                (stats,),
            )
            cur.execute(
                "SELECT (feature_stats -> 'price_delta' ->> 'psi_recent_30d')::float "
                "FROM model_scorecards WHERE symbol = 'ALL' AND feature_stats IS NOT NULL"
            )
            psi = cur.fetchone()
            cur.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'model_scorecards' AND column_name = 'feature_stats'"
            )
            data_type = cur.fetchone()

        assert psi == (pytest.approx(0.31),)
        assert data_type == ("jsonb",)
    # -- Migracja 055: znacznik nieaktualnego notowania (DAT-14) ---------------

    def test_stale_quote_column_is_a_nullable_boolean(self, pg_conn) -> None:
        """Zapis bez flagi `stale_quote_enabled` zostawia NULL (nieznane); reflect
        stawia true przy rozliczeniu z notowania sprzed sesji po predykcji."""
        assert "stale_quote" in _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
                "VALUES ('SQ_NULL', '2026-11-26T05:30:00+00:00', 100.0) RETURNING stale_quote"
            )
            assert cur.fetchone() == (None,)
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction, stale_quote) "
                "VALUES ('SQ_SET', '2026-11-26T05:30:00+00:00', 100.0, false) RETURNING id"
            )
            prediction_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE prediction_logs SET stale_quote = true, actual_price_after_12h = 100.0 "
                "WHERE id = %s RETURNING stale_quote",
                (prediction_id,),
            )
            assert cur.fetchone() == (True,)

    # -- Migracja 056: kapitalizacja w cache fundamentów (DAT-21) --------------

    def test_fundamentals_cache_market_cap_is_a_nullable_number(self, pg_conn) -> None:
        """Upsert bez flagi zostawia NULL (próg 100k USD); z flagą trzyma pełne USD."""
        assert "market_cap" in _columns(pg_conn, "fundamentals_cache")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO fundamentals_cache (symbol, trailing_pe) VALUES ('MC_NULL', 30) "
                "RETURNING market_cap"
            )
            assert cur.fetchone() == (None,)
            cur.execute(
                "INSERT INTO fundamentals_cache (symbol, market_cap) "
                "VALUES ('MC_SET', 4123456789012) RETURNING market_cap"
            )
            assert int(cur.fetchone()[0]) == 4123456789012

    # -- Migracja 057: alerty poziomów cenowych (ALR-11) ------------------------

    def test_price_level_alerts_keep_one_row_per_level_and_direction(self, pg_conn) -> None:
        """Lista startowa z config.toml dopisuje się upsertem ON CONFLICT DO NOTHING —
        poziom wyłączony po trafieniu zostaje wyłączony."""
        assert {
            "id", "symbol", "level", "direction", "note", "active",
            "last_seen_price", "triggered_at", "created_at",
        } <= _columns(pg_conn, "price_level_alerts")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO price_level_alerts (symbol, level, direction, note) "
                "VALUES ('PL_NVDA', 150, 'ABOVE', 'realizacja części zysku') "
                "RETURNING active, last_seen_price, triggered_at"
            )
            assert cur.fetchone() == (True, None, None)
            cur.execute(
                "UPDATE price_level_alerts SET active = false, triggered_at = NOW() "
                "WHERE symbol = 'PL_NVDA'"
            )
            cur.execute(
                "INSERT INTO price_level_alerts (symbol, level, direction) "
                "VALUES ('PL_NVDA', 150.0, 'ABOVE'), ('PL_NVDA', 150, 'BELOW') "
                "ON CONFLICT (symbol, level, direction) DO NOTHING"
            )
            cur.execute(
                "SELECT direction, active FROM price_level_alerts "
                "WHERE symbol = 'PL_NVDA' ORDER BY direction"
            )
            assert cur.fetchall() == [("ABOVE", False), ("BELOW", True)]
        assert "price_level_alerts_active_idx" in _indexes(pg_conn, "price_level_alerts")

    @pytest.mark.parametrize(
        ("level", "direction"), [("0", "ABOVE"), ("-5", "BELOW"), ("150", "SIDEWAYS")]
    )
    def test_price_level_alerts_reject_a_bad_level_or_direction(
        self, pg_conn, level: str, direction: str
    ) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO price_level_alerts (symbol, level, direction) "
                "VALUES ('PL_BAD', %s, %s)",
                (level, direction),
            )

    # -- Migracja 038: RLS na każdej tabeli public (BEZ-05) ---------------------

    def test_every_public_table_has_row_level_security(self, pg_conn) -> None:
        """Przy domyślnych grantach Supabase tabela `public` bez RLS jest czytelna
        i zapisywalna kluczem publicznym (anon). 038 włącza RLS na wszystkich, a
        tabela dodana w kolejnej migracji musi włączyć je sama — inaczej ten test
        padnie. Biegnie po `test_migrations_are_idempotent`, więc sprawdza też, że
        ponowny przebieg 001/002 (jawne DISABLE) kończy się z RLS włączonym."""
        with pg_conn.cursor() as cur:
            cur.execute(_PUBLIC_TABLES_SQL)
            row_security = dict(cur.fetchall())

        # Strażnik pustego wyniku: bez migracji lista byłaby pusta i test zielony.
        assert {"subscribers", "positions", "prediction_logs"} <= row_security.keys()
        assert sorted(name for name, enabled in row_security.items() if not enabled) == []


# -- TEST-15: emulacja ról Supabase dla testów uprawnień ----------------------

# Role API jak w obrazie Supabase: klucz publiczny (anon / publishable) działa
# jako `anon`, JWT zalogowanego użytkownika jako `authenticated`, klucz secret /
# service_role jako `service_role`, który omija RLS (BYPASSRLS).
SUPABASE_API_ROLES = {
    "anon": "NOLOGIN NOINHERIT",
    "authenticated": "NOLOGIN NOINHERIT",
    "service_role": "NOLOGIN NOINHERIT BYPASSRLS",
}
PUBLIC_API_ROLES = ("anon", "authenticated")
SUPABASE_LIKE_DB = "supabase_like"

# Domyślne uprawnienia, które Supabase ustawia roli tworzącej obiekty w `public`
# (tam `postgres`, tu superuser kontenera aplikujący migracje): każda nowa
# tabela, widok, sekwencja i funkcja trafia do wszystkich trzech ról API.
SUPABASE_DEFAULT_GRANTS = """
    GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT ALL ON TABLES TO anon, authenticated, service_role;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT ALL ON SEQUENCES TO anon, authenticated, service_role;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT ALL ON FUNCTIONS TO anon, authenticated, service_role;
"""

# Pełna lista uprawnień tabeli w Postgres 16 (MAINTAIN dochodzi dopiero w 17).
ANY_TABLE_PRIVILEGE = "SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER"
ANY_SEQUENCE_PRIVILEGE = "USAGE, SELECT, UPDATE"

# Obiekty `public` należące do rozszerzeń (pgvector, pgcrypto) pomijamy: ich
# funkcje mają EXECUTE dla PUBLIC z definicji rozszerzenia i nie niosą danych.
_NOT_EXTENSION_MEMBER = """
    NOT EXISTS (
        SELECT 1 FROM pg_depend d
        WHERE d.classid = '{catalog}'::regclass AND d.objid = {oid} AND d.deptype = 'e'
    )
"""
_APP_RELATIONS_SQL = f"""
    SELECT c.relname, c.relkind
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
      AND {_NOT_EXTENSION_MEMBER.format(catalog="pg_class", oid="c.oid")}
"""
_APP_FUNCTIONS_SQL = f"""
    SELECT p.oid::regprocedure::text
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public'
      AND p.prokind = 'f'
      AND {_NOT_EXTENSION_MEMBER.format(catalog="pg_proc", oid="p.oid")}
"""


@pytest.fixture(scope="module")
def supabase_like_conn(pg_conn):
    """Osobna baza w TYM SAMYM kontenerze co `pg_conn` (drugi kontener to kolejne
    setki MB przy pełnym swapie): najpierw role i domyślne granty jak w Supabase,
    potem komplet migracji — w tej samej kolejności co na produkcji.

    Superuser kontenera gra rolę `postgres` (właściciel obiektów, omija RLS),
    a testy przełączają się na role API przez `SET LOCAL ROLE`. Role są
    obiektami klastra, więc widzi je też baza `pg_conn`; `TestMigrations` biegnie
    wcześniej (kolejność w pliku), a guard pg_roles w 038 działa w obu stanach.
    To emulacja, nie Supabase: brak PostgREST, JWT i Postgresa 17 (MAINTAIN)."""
    with pg_conn.cursor() as cur:
        for role, attributes in SUPABASE_API_ROLES.items():
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE ROLE {} {}").format(
                        sql.Identifier(role), sql.SQL(attributes)
                    )
                )
        cur.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(SUPABASE_LIKE_DB))
        )
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(SUPABASE_LIKE_DB)))

    info = pg_conn.info
    conn = psycopg2.connect(
        host=info.host,
        port=info.port,
        dbname=SUPABASE_LIKE_DB,
        user=info.user,
        password=info.password,
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(SUPABASE_DEFAULT_GRANTS)
    _apply_migrations(conn)
    yield conn
    conn.close()


@contextmanager
def _rolled_back(
    conn: psycopg2.extensions.connection,
) -> Iterator[psycopg2.extensions.cursor]:
    """Transakcja zawsze wycofywana: dane testu, granty i `SET LOCAL ROLE` znikają
    razem z nią, więc kolejny test zaczyna jako superuser na czystej bazie."""
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            yield cur
        finally:
            cur.execute("ROLLBACK")


def _set_role(cur: psycopg2.extensions.cursor, role: str) -> None:
    cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))


def _has_table_privilege(cur: psycopg2.extensions.cursor, role: str, relation: str) -> bool:
    cur.execute(
        "SELECT has_table_privilege(%s::name, %s::regclass, %s)",
        (role, relation, ANY_TABLE_PRIVILEGE),
    )
    return bool(cur.fetchone()[0])


@pytest.mark.containers
class TestSupabaseRoleEmulation:
    """Test migracji jako superuser nie widzi regresji uprawnień: superuser omija
    i RLS, i granty. Tu sprawdzamy bazę oczami ról, którymi łączy się PostgREST —
    po 038 klucz publiczny (anon) nic nie widzi, a klucz secret (service_role)
    dalej czyta, pisze i woła RPC aplikacji."""

    def test_anon_cannot_read_subscribers(self, supabase_like_conn) -> None:
        """Bez 038 domyślne granty Supabase i wyłączony RLS oddają anon e-mail."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute("INSERT INTO subscribers (email) VALUES ('reader@example.com')")
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT email FROM subscribers")

    def test_anon_cannot_read_positions(self, supabase_like_conn) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost) VALUES ('NVDA', 10, 120.5)"
            )
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT symbol, quantity, avg_cost FROM positions")

    def test_anon_select_is_denied_on_every_relation(self, supabase_like_conn) -> None:
        """Każda tabela, widok i `ml_feature_store` — prawdziwy SELECT, nie katalog."""
        with supabase_like_conn.cursor() as cur:
            cur.execute(_APP_RELATIONS_SQL)
            relations = [name for name, kind in cur.fetchall() if kind != "S"]
        assert {"subscribers", "positions", "ml_feature_store"} <= set(relations)

        readable = []
        for relation in relations:
            with _rolled_back(supabase_like_conn) as cur:
                _set_role(cur, "anon")
                try:
                    cur.execute(
                        sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(relation))
                    )
                except psycopg2.errors.InsufficientPrivilege:
                    continue
                readable.append(relation)
        assert readable == []

    @pytest.mark.parametrize("role", PUBLIC_API_ROLES)
    def test_public_roles_hold_no_privilege_on_any_relation(
        self, supabase_like_conn, role: str
    ) -> None:
        """Tabele, widoki, widok zmaterializowany i sekwencje — także INSERT,
        UPDATE, DELETE i TRUNCATE, nie tylko odczyt."""
        with supabase_like_conn.cursor() as cur:
            cur.execute(_APP_RELATIONS_SQL)
            relations = cur.fetchall()
            granted = []
            for name, kind in relations:
                if kind == "S":
                    cur.execute(
                        "SELECT has_sequence_privilege(%s::name, %s::regclass, %s)",
                        (role, name, ANY_SEQUENCE_PRIVILEGE),
                    )
                    held = bool(cur.fetchone()[0])
                else:
                    held = _has_table_privilege(cur, role, name)
                if held:
                    granted.append(name)

        assert "council_votes_id_seq" in {name for name, _ in relations}
        assert granted == []

    @pytest.mark.parametrize("role", PUBLIC_API_ROLES)
    def test_public_roles_cannot_execute_app_functions(
        self, supabase_like_conn, role: str
    ) -> None:
        """EXECUTE z domyślnego grantu roli PUBLIC też się liczy — `anon` jest jej
        członkiem. Nowa funkcja w kolejnej migracji musi odebrać go sama."""
        with supabase_like_conn.cursor() as cur:
            cur.execute(_APP_FUNCTIONS_SQL)
            functions = [row[0] for row in cur.fetchall()]
            executable = []
            for function in functions:
                cur.execute(
                    "SELECT has_function_privilege(%s::name, %s::regprocedure, 'EXECUTE')",
                    (role, function),
                )
                if cur.fetchone()[0]:
                    executable.append(function)

        assert any(name.startswith("match_news_embeddings(") for name in functions)
        assert executable == []

    def test_rls_hides_every_row_even_if_a_grant_comes_back(self, supabase_like_conn) -> None:
        """Druga warstwa: gdy grant wróci (przyszła migracja, ręczny GRANT w
        dashboardzie), RLS bez polityk dalej zwraca anon zero wierszy."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute("INSERT INTO subscribers (email) VALUES ('leak@example.com')")
            cur.execute("GRANT SELECT ON subscribers TO anon")
            _set_role(cur, "anon")
            cur.execute("SELECT count(*) FROM subscribers")
            assert cur.fetchone() == (0,)

    def test_objects_created_after_the_migration_are_not_granted_to_public_roles(
        self, supabase_like_conn
    ) -> None:
        """ALTER DEFAULT PRIVILEGES z 038: nowa tabela albo DROP + CREATE widoku
        (wzorzec 026/027) nie przywracają grantów anon / authenticated, a
        service_role dalej je dostaje."""
        probes = ("probe_table", "probe_view", "probe_matview")
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute("CREATE TABLE probe_table (id BIGSERIAL PRIMARY KEY)")
            cur.execute("CREATE VIEW probe_view AS SELECT 1 AS one")
            cur.execute("CREATE MATERIALIZED VIEW probe_matview AS SELECT 1 AS one")

            granted = [
                (role, probe)
                for role in PUBLIC_API_ROLES
                for probe in probes
                if _has_table_privilege(cur, role, probe)
            ]
            cur.execute(
                "SELECT has_sequence_privilege(%s::name, 'probe_table_id_seq'::regclass, %s)",
                ("anon", ANY_SEQUENCE_PRIVILEGE),
            )
            anon_sequence = bool(cur.fetchone()[0])
            service_role = [_has_table_privilege(cur, "service_role", probe) for probe in probes]

        assert granted == []
        assert anon_sequence is False
        assert service_role == [True, True, True]

    def test_service_role_reads_and_writes_through_rls(self, supabase_like_conn) -> None:
        """Klucz secret / service_role po 038: PII i portfel, predykcja z
        rozliczeniem (trigger 029), głos rady (BIGSERIAL → USAGE sekwencji),
        upsert snapshotu ceny i odczyt `ml_feature_store`."""
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "service_role")

            cur.execute("INSERT INTO subscribers (email) VALUES ('owner@example.com')")
            cur.execute("SELECT email FROM subscribers")
            assert ("owner@example.com",) in cur.fetchall()

            cur.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost) VALUES ('NVDA', 10, 120.5)"
            )
            cur.execute("SELECT symbol, currency FROM positions")
            assert ("NVDA", "USD") in cur.fetchall()

            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction) "
                "VALUES ('SR_NVDA', 100.0) RETURNING id"
            )
            prediction_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 105.0 WHERE id = %s "
                "RETURNING settled_at IS NOT NULL",
                (prediction_id,),
            )
            assert cur.fetchone() == (True,)

            cur.execute(
                "INSERT INTO council_votes "
                "(prediction_id, symbol, investor_name, recommendation, confidence) "
                "VALUES (%s, 'SR_NVDA', 'Buffett', 'BUY', 0.8) RETURNING id",
                (prediction_id,),
            )
            assert cur.fetchone() is not None

            for price in (100.0, 101.0):
                cur.execute(
                    "INSERT INTO price_snapshots (symbol, price, timestamp) "
                    "VALUES ('SR_NVDA', %s, '2026-10-20T10:15:00+00:00') "
                    "ON CONFLICT (symbol, timestamp_hour) DO UPDATE SET price = EXCLUDED.price",
                    (price,),
                )
            cur.execute("SELECT price FROM price_snapshots WHERE symbol = 'SR_NVDA'")
            assert [float(price) for (price,) in cur.fetchall()] == [101.0]

            # Widok sprzed REFRESH: pusty, ale odczyt jest dozwolony.
            cur.execute("SELECT count(*) FROM ml_feature_store")
            assert cur.fetchone() == (0,)

    def test_feature_store_rebuilt_by_039_stays_closed_to_public_roles(
        self, supabase_like_conn
    ) -> None:
        """039 robi DROP + CREATE widoku PO 038: anon / authenticated bez żadnego
        prawa, service_role dalej czyta (Slow Loop trenuje z tego widoku)."""
        with _rolled_back(supabase_like_conn) as cur:
            public = [
                _has_table_privilege(cur, role, "ml_feature_store") for role in PUBLIC_API_ROLES
            ]
            cur.execute(
                "SELECT has_table_privilege('service_role', 'ml_feature_store', 'SELECT')"
            )
            service_role = cur.fetchone()[0]

        assert public == [False, False]
        assert service_role is True

    def test_llm_calls_is_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        """FIN-13: nowa tabela po 038 — RLS włączony w 040, anon i authenticated bez
        prawa, a service_role (Fast i Slow Loop) zapisuje batch i czyta."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'llm_calls'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [_has_table_privilege(cur, role, "llm_calls") for role in PUBLIC_API_ROLES]

            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO llm_calls (run_id, symbol, node, provider, model, input_tokens, "
                "output_tokens, cost_usd, price_as_of) VALUES "
                "('LC_SR', 'NVDA', 'predict', 'anthropic', 'claude-sonnet-4-6', 1000, 200, "
                "0.006, '2026-10-20'), "
                "('LC_SR', NULL, 'weekly_recap', 'openai', 'gpt-5-mini', 10, 5, NULL, NULL)"
            )
            cur.execute("SELECT count(*) FROM llm_calls WHERE run_id = 'LC_SR'")
            assert cur.fetchone() == (2,)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM llm_calls LIMIT 1")

    def test_cycle_runs_is_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        """NIEZ-29: nowa tabela po 038 — RLS włączony w 041, anon i authenticated bez
        prawa, a service_role (Fast Loop) zapisuje przebieg i czyta historię."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'cycle_runs'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [_has_table_privilege(cur, role, "cycle_runs") for role in PUBLIC_API_ROLES]

            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO cycle_runs (loop, run_id, run_attempt, started_at, duration_s, "
                "eligible, saved, failures, paid_calls, report_delivered) VALUES "
                "('fast', 'CR_SR', 1, NOW(), 312.5, 40, 6, 1, "
                "'{\"llm\": 7, \"council_llm\": 48}', true)"
            )
            cur.execute("SELECT paid_calls ->> 'llm' FROM cycle_runs WHERE run_id = 'CR_SR'")
            assert cur.fetchone() == ("7",)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM cycle_runs LIMIT 1")

    def test_reference_predictions_are_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        """WIA-24: nowa tabela po 038 — RLS włączony w 042, anon i authenticated bez
        prawa, a service_role (Slow Loop) zapisuje odczyt i rozlicza wywołanie."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class "
                "WHERE oid = 'reference_predictions'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [
                _has_table_privilege(cur, role, "reference_predictions")
                for role in PUBLIC_API_ROLES
            ]

            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO reference_predictions (symbol, read_at, price, predicted_trend) "
                "VALUES ('REF_SR', '2026-10-18T03:00:00Z', 100, 'BEARISH')"
            )
            cur.execute(
                "UPDATE reference_predictions SET outcome = 'MISS', settled_price = 101, "
                "settled_at = '2026-10-25T03:00:00Z' WHERE symbol = 'REF_SR'"
            )
            cur.execute("SELECT outcome FROM reference_predictions WHERE symbol = 'REF_SR'")
            assert cur.fetchone() == ("MISS",)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM reference_predictions LIMIT 1")

    def test_price_level_alerts_are_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        """ALR-11: nowa tabela po 038 — RLS włączony w 057, anon i authenticated bez
        prawa, a service_role (watch) dopisuje poziom, czyta aktywne i wyłącza trafiony."""
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'price_level_alerts'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [
                _has_table_privilege(cur, role, "price_level_alerts") for role in PUBLIC_API_ROLES
            ]

            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO price_level_alerts (symbol, level, direction) "
                "VALUES ('PL_SR', 150, 'ABOVE') RETURNING id"
            )
            alert_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE price_level_alerts SET active = false, last_seen_price = 151.2, "
                "triggered_at = NOW() WHERE id = %s",
                (alert_id,),
            )
            cur.execute("SELECT count(*) FROM price_level_alerts WHERE symbol = 'PL_SR' AND active")
            assert cur.fetchone() == (0,)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM price_level_alerts LIMIT 1")

    def test_service_role_executes_the_rpc_functions(self, supabase_like_conn) -> None:
        """RPC, które aplikacja woła przez PostgREST. `refresh_ml_feature_store`
        poza zakresem: REFRESH wymaga właściciela widoku, a prawo MAINTAIN dla
        grantu ALL przynosi dopiero Postgres 17."""
        vector = "[" + ",".join(["0.1"] * 1536) + "]"
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "service_role")
            # Baza emulacji nie ma zatwierdzonych wierszy — liczy się brak 42501.
            cur.execute("SELECT count(*) FROM match_news_embeddings(%s::vector, 3)", (vector,))
            assert cur.fetchone() == (0,)
            cur.execute("SELECT count(*) FROM persona_accuracy_weights(90)")
            assert cur.fetchone() == (0,)
            cur.execute("SELECT count(*) FROM persona_accuracy_stats(90)")
            assert cur.fetchone() == (0,)

    def test_service_role_executes_persona_vote_outcomes(self, supabase_like_conn) -> None:
        """WIA-51: RPC 045 woła aplikacja — EXECUTE tylko dla service_role."""
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "service_role")
            cur.execute("SELECT count(*) FROM persona_vote_outcomes(90)")
            assert cur.fetchone() == (0,)

    def test_service_role_executes_persona_scoring_stats(self, supabase_like_conn) -> None:
        """WIA-44: RPC 065 woła aplikacja — EXECUTE tylko dla service_role."""
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "service_role")
            cur.execute("SELECT count(*) FROM persona_scoring_stats(90, ARRAY['BTC'])")
            assert cur.fetchone() == (0,)


# BEZ-18: RPC aplikacji miały zmienny search_path (Supabase advisor
# `function_search_path_mutable`). Wywołanie z innej roli i innym search_path mogłoby
# podstawić obiekt o tej samej nazwie. EXECUTE odebrane PUBLIC i rolom API już w 038.
RPC_FUNCTIONS = (
    "match_news_embeddings(vector,integer,text)",
    "persona_accuracy_weights(integer)",
    "persona_accuracy_stats(integer)",
)
_PUBLIC_GRANTEE = 0


@pytest.mark.containers
class TestRpcFunctionHardening:
    @pytest.mark.parametrize("function", RPC_FUNCTIONS)
    def test_rpc_function_pins_its_search_path(self, supabase_like_conn, function: str) -> None:
        with supabase_like_conn.cursor() as cur:
            cur.execute("SELECT proconfig FROM pg_proc WHERE oid = %s::regprocedure", (function,))
            (config,) = cur.fetchone()

        assert any(item.startswith("search_path=") for item in config or []), config

    @pytest.mark.parametrize("function", RPC_FUNCTIONS)
    def test_rpc_functions_not_executable_by_arbitrary_role(
        self, supabase_like_conn, function: str
    ) -> None:
        """PUBLIC (`acldefault`, gdy `proacl` jest NULL) i role API nie wykonują RPC;
        klucz secret (service_role) tak."""
        with supabase_like_conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_proc p,
                         aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) acl
                    WHERE p.oid = %s::regprocedure
                      AND acl.grantee = %s
                      AND acl.privilege_type = 'EXECUTE'
                )
                """,
                (function, _PUBLIC_GRANTEE),
            )
            public_can_execute = cur.fetchone()[0]
            can_execute = {}
            for role in (*PUBLIC_API_ROLES, "service_role"):
                cur.execute(
                    "SELECT has_function_privilege(%s::name, %s::regprocedure, 'EXECUTE')",
                    (role, function),
                )
                can_execute[role] = cur.fetchone()[0]

        assert public_can_execute is False
        assert can_execute == {"anon": False, "authenticated": False, "service_role": True}


def _schema_version_function_count(cur: psycopg2.extensions.cursor) -> int:
    cur.execute(
        """
        SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'schema_version'
        """
    )
    return int(cur.fetchone()[0])


def _create_migration_registry(cur: psycopg2.extensions.cursor, versions: Sequence[str]) -> None:
    """Rejestr, który tworzy Supabase CLI (`supabase db push`); w emulacji go nie ma."""
    cur.execute("CREATE SCHEMA supabase_migrations")
    cur.execute("CREATE TABLE supabase_migrations.schema_migrations (version text PRIMARY KEY)")
    for version in versions:
        cur.execute(
            "INSERT INTO supabase_migrations.schema_migrations (version) VALUES (%s)", (version,)
        )


@pytest.mark.containers
class TestSchemaVersionFunction:
    """NIEZ-30: `public.schema_version()` mówi aplikacji, do której migracji sięga baza,
    żeby flaga bez swojej migracji wyłączyła się sama zamiast kończyć się PGRST204."""

    def test_is_a_security_definer_with_a_fixed_search_path(self, supabase_like_conn) -> None:
        with supabase_like_conn.cursor() as cur:
            assert _schema_version_function_count(cur) == 1
            cur.execute(
                """
                SELECT p.prosecdef, p.proconfig, p.prorettype::regtype::text
                FROM pg_proc p WHERE p.oid = 'public.schema_version()'::regprocedure
                """
            )
            security_definer, config, return_type = cur.fetchone()

        assert security_definer is True
        assert any(item.startswith("search_path=") for item in config or []), config
        assert return_type == "integer"

    def test_is_null_without_the_registry_or_with_an_empty_one(self, supabase_like_conn) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            assert _schema_version_function_count(cur) == 1
            cur.execute("SELECT public.schema_version()")
            without_registry = cur.fetchone()
            _create_migration_registry(cur, [])
            cur.execute("SELECT public.schema_version()")
            empty_registry = cur.fetchone()

        assert without_registry == (None,)
        assert empty_registry == (None,)

    def test_reads_the_highest_numeric_version_and_ignores_timestamp_versions(
        self, supabase_like_conn
    ) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            assert _schema_version_function_count(cur) == 1
            _create_migration_registry(cur, ["001", "059", "070", "20240101000000"])
            cur.execute("SELECT public.schema_version()")
            version = cur.fetchone()

        assert version == (70,)

    def test_service_role_reads_it_without_access_to_the_registry(self, supabase_like_conn) -> None:
        """SECURITY DEFINER: klucz secret nie ma USAGE na schemacie supabase_migrations."""
        with _rolled_back(supabase_like_conn) as cur:
            assert _schema_version_function_count(cur) == 1
            _create_migration_registry(cur, ["001", "059"])
            _set_role(cur, "service_role")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT version FROM supabase_migrations.schema_migrations")
        with _rolled_back(supabase_like_conn) as cur:
            _create_migration_registry(cur, ["001", "059"])
            _set_role(cur, "service_role")
            cur.execute("SELECT public.schema_version()")
            version = cur.fetchone()

        assert version == (59,)

    @pytest.mark.parametrize("role", PUBLIC_API_ROLES)
    def test_public_api_roles_cannot_execute_it(self, supabase_like_conn, role: str) -> None:
        with supabase_like_conn.cursor() as cur:
            assert _schema_version_function_count(cur) == 1
            cur.execute(
                "SELECT has_function_privilege(%s::name, 'public.schema_version()'::regprocedure, "
                "'EXECUTE')",
                (role,),
            )
            executable = cur.fetchone()[0]

        assert executable is False


@pytest.fixture
def fresh_pg():
    """Świeży, IZOLOWANY kontener Postgres (function scope) — dla testów, które
    muszą zacząć od stanu PRZED konkretną migracją (nie da się tego uzyskać w
    module-scoped kontenerze, gdzie wszystkie migracje są już zaaplikowane)."""
    with _pgvector_connection() as conn:
        yield conn


@pytest.mark.containers
class TestMigration013DedupesPreexistingDuplicates:
    """Regresja: 013 zakładał czystą tabelę i robił od razu CREATE UNIQUE INDEX.
    Na produkcji `price_snapshots`/`prediction_logs` miały już DUPLIKATY z ery
    przed idempotencją (gołe .insert() przy re-run/retry), więc unique index
    wywalał się błędem 23505. Migracja MUSI najpierw zdeduplikować."""

    def _apply_up_to_012(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("013_idempotency_and_pagination.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def test_013_dedupes_then_builds_unique_index(self, fresh_pg) -> None:
        conn = fresh_pg
        self._apply_up_to_012(conn)
        with conn.cursor() as cur:
            # Dwa snapshoty TEJ SAMEJ godziny (17:10 i 17:40 → hour 17:00) =
            # duplikat (symbol, timestamp_hour). Odtwarza błąd z produkcji.
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) VALUES "
                "('VT', 100.0, '2026-06-03 17:10:00+00'), "
                "('VT', 101.0, '2026-06-03 17:40:00+00')"
            )
            # prediction_logs: rozliczona (17:10) vs nierozliczona (17:40) —
            # dedup MUSI zachować rozliczoną (nie tracimy accuracy/feedbacku).
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, predicted_trend, is_trend_correct) VALUES "
                "('VT', '2026-06-03 17:10:00+00', 'BULLISH', true), "
                "('VT', '2026-06-03 17:40:00+00', 'BEARISH', NULL)"
            )

            # Aplikacja 013 NIE może rzucić mimo duplikatów (sedno naprawy).
            cur.execute(
                (MIGRATIONS_DIR / "013_idempotency_and_pagination.sql").read_text()
            )

            # Unikalne indeksy powstały.
            assert "idx_price_snapshots_symbol_hour_uniq" in _indexes(
                conn, "price_snapshots"
            )
            assert "idx_prediction_logs_symbol_hour_uniq" in _indexes(
                conn, "prediction_logs"
            )

            # Zdeduplikowane: jeden wiersz na (symbol, godzina).
            cur.execute("SELECT count(*) FROM price_snapshots WHERE symbol='VT'")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM prediction_logs WHERE symbol='VT'")
            assert cur.fetchone()[0] == 1

            # price_snapshots: zachowany NAJNOWSZY snapshot (101 @ 17:40).
            cur.execute("SELECT price FROM price_snapshots WHERE symbol='VT'")
            assert float(cur.fetchone()[0]) == pytest.approx(101.0)
            # prediction_logs: zachowany ROZLICZONY wiersz (BULLISH @ 17:10),
            # mimo że jest starszy — feedback ma priorytet nad świeżością.
            cur.execute(
                "SELECT predicted_trend, is_trend_correct "
                "FROM prediction_logs WHERE symbol='VT'"
            )
            kept = cur.fetchone()
            assert kept[0] == "BULLISH"
            assert kept[1] is True


@pytest.mark.containers
class TestMigration027BackfillsFeatureSetVersion:
    """DAT-09: wiersze sprzed okna 24 h (3738ef2, 2026-10-18 21:47:22 UTC) to
    pewna wersja 1; późniejsze bez stempla zostają NULL, bo moment wdrożenia
    nie jest zapisany w bazie i taki wiersz mógł powstać ze starego kodu."""

    def _apply_up_to_026(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("027_feature_set_version.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def test_027_backfills_only_rows_before_the_news_window_change(
        self, fresh_pg
    ) -> None:
        conn = fresh_pg
        self._apply_up_to_026(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h) "
                "VALUES "
                "('FSV_OLD', '2026-10-18T21:47:21+00:00', 100.0, 101.0), "
                "('FSV_NEW', '2026-10-18T21:47:22+00:00', 100.0, 101.0)"
            )

            cur.execute((MIGRATIONS_DIR / "027_feature_set_version.sql").read_text())

            cur.execute(
                "SELECT symbol, feature_set_version FROM prediction_logs "
                "WHERE symbol LIKE 'FSV_%'"
            )
            versions = dict(cur.fetchall())

        assert versions == {"FSV_OLD": 1, "FSV_NEW": None}


@pytest.mark.containers
class TestMigration029BackfillsSettledAt:
    """WIA-19: chwila rozliczenia sprzed triggera nie jest nigdzie zapisana.
    Cykl, który rozlicza predykcję, zapisuje snapshot ceny po tej samej cenie,
    więc najlepszym przybliżeniem jest pierwszy taki snapshot po cyklu predykcji;
    bez niego zostaje dolna granica — `timestamp` predykcji."""

    def _apply_up_to_028(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("029_prediction_settled_at.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def test_029_backfills_settled_at_from_the_settling_cycle_snapshot(
        self, fresh_pg
    ) -> None:
        conn = fresh_pg
        self._apply_up_to_028(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, actual_price_after_12h) "
                "VALUES "
                # Piątek → rozliczona w poniedziałek po 105.
                "('BF_FRI', '2026-10-16T17:30:00+00:00', 100.0, 105.0), "
                # Cena bez zmian: snapshot cyklu predykcji (ta sama cena) się nie liczy.
                "('BF_FLAT', '2026-10-16T17:30:00+00:00', 100.0, 100.0), "
                # Brak snapshotu po cenie rozliczenia → dolna granica.
                "('BF_NONE', '2026-10-16T17:30:00+00:00', 100.0, 107.0), "
                "('BF_OPEN', '2026-10-16T17:30:00+00:00', 100.0, NULL)"
            )
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) VALUES "
                "('BF_FRI', 100.0, '2026-10-16T17:30:05+00:00'), "
                "('BF_FRI', 103.0, '2026-10-17T05:30:05+00:00'), "
                "('BF_FRI', 105.0, '2026-10-19T05:30:05+00:00'), "
                "('BF_FRI', 105.0, '2026-10-19T17:30:05+00:00'), "
                "('BF_FLAT', 100.0, '2026-10-16T17:30:05+00:00'), "
                "('BF_FLAT', 100.0, '2026-10-19T05:30:05+00:00'), "
                "('BF_NONE', 100.0, '2026-10-16T17:30:05+00:00'), "
                "('BF_NONE', 106.0, '2026-10-19T05:30:05+00:00')"
            )

            cur.execute((MIGRATIONS_DIR / "029_prediction_settled_at.sql").read_text())

            cur.execute(
                "SELECT symbol, settled_at FROM prediction_logs WHERE symbol LIKE 'BF_%'"
            )
            settled = {symbol: stamp for symbol, stamp in cur.fetchall()}

        monday = datetime(2026, 10, 19, 5, 30, 5, tzinfo=UTC)
        assert settled == {
            "BF_FRI": monday,
            "BF_FLAT": monday,
            "BF_NONE": datetime(2026, 10, 16, 17, 30, tzinfo=UTC),
            "BF_OPEN": None,
        }


@pytest.mark.containers
class TestMigration034BackfillsGateDelta:
    """EXP-01: bramka liczyła deltę względem NAJNOWSZEGO snapshotu ceny z chwili
    startu cyklu, a snapshot cyklu predykcji powstaje dopiero w węźle
    terminalnym, PO zapisie predykcji. Dlatego backfill bierze najnowszy
    snapshot symbolu sprzed `timestamp` predykcji; bez niego (albo z ceną 0)
    zostaje NULL."""

    def _apply_up_to_033(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("034_gate_delta.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def _gate_deltas(self, cur: psycopg2.extensions.cursor) -> dict[str, float | None]:
        cur.execute("SELECT symbol, gate_delta FROM prediction_logs WHERE symbol LIKE 'GB_%'")
        return dict(cur.fetchall())

    def test_034_backfills_the_delta_against_the_previous_cycle_snapshot(
        self, fresh_pg
    ) -> None:
        conn = fresh_pg
        self._apply_up_to_033(conn)
        migration = (MIGRATIONS_DIR / "034_gate_delta.sql").read_text()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) VALUES "
                "('GB_PREV', '2026-10-16T17:30:00+00:00', 103.1), "
                "('GB_FIRST', '2026-10-16T17:30:00+00:00', 50.0), "
                "('GB_ZERO', '2026-10-16T17:30:00+00:00', 10.0)"
            )
            cur.execute(
                "INSERT INTO price_snapshots (symbol, price, timestamp) VALUES "
                # Starszy i najnowszy snapshot sprzed predykcji — liczy się najnowszy.
                "('GB_PREV', 90.0, '2026-10-15T17:30:05+00:00'), "
                "('GB_PREV', 100.0, '2026-10-16T05:30:05+00:00'), "
                # Snapshot cyklu predykcji (po zapisie predykcji) nie jest referencją.
                "('GB_PREV', 103.1, '2026-10-16T17:30:05+00:00'), "
                "('GB_FIRST', 50.0, '2026-10-16T17:30:05+00:00'), "
                "('GB_ZERO', 0.0, '2026-10-16T05:30:05+00:00')"
            )

            cur.execute(migration)
            deltas = self._gate_deltas(cur)

            assert deltas["GB_PREV"] == pytest.approx(0.031)
            assert deltas["GB_FIRST"] is None
            assert deltas["GB_ZERO"] is None

            # Idempotencja: ponowny przebieg nie nadpisuje zapisanej wartości.
            cur.execute("UPDATE prediction_logs SET gate_delta = 0.5 WHERE symbol = 'GB_PREV'")
            cur.execute(migration)
            assert self._gate_deltas(cur)["GB_PREV"] == pytest.approx(0.5)


@pytest.mark.containers
class TestMigration035BackfillsPositionCurrency:
    """PORT-18: waluta istniejących pozycji z sufiksu tickera — .DE/.AS/.PA/.MI/.MC
    → EUR, .L → GBP (bez przeliczania GBX), reszta → USD. Data zakupu zostaje NULL,
    bo `as_of` to data aktualizacji tabeli, nie zakupu."""

    def _apply_up_to_034(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("035_positions_currency.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def _currencies(self, cur: psycopg2.extensions.cursor) -> dict[str, str]:
        cur.execute("SELECT symbol, currency FROM positions")
        return dict(cur.fetchall())

    def test_035_derives_the_currency_from_the_ticker_suffix(self, fresh_pg) -> None:
        conn = fresh_pg
        self._apply_up_to_034(conn)
        migration = (MIGRATIONS_DIR / "035_positions_currency.sql").read_text()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO positions (symbol, quantity, avg_cost) VALUES "
                "('SAP.DE', 1, 1), ('ASML.AS', 1, 1), ('AIR.PA', 1, 1), "
                "('ENI.MI', 1, 1), ('ITX.MC', 1, 1), ('VUSA.L', 1, 1), "
                "('xdwd.de', 1, 1), ('AAPL', 1, 1), ('BRK.B', 1, 1)"
            )

            cur.execute(migration)

            assert self._currencies(cur) == {
                "SAP.DE": "EUR",
                "ASML.AS": "EUR",
                "AIR.PA": "EUR",
                "ENI.MI": "EUR",
                "ITX.MC": "EUR",
                "VUSA.L": "GBP",
                "xdwd.de": "EUR",
                "AAPL": "USD",
                "BRK.B": "USD",
            }
            cur.execute("SELECT count(*) FROM positions WHERE purchase_date IS NOT NULL")
            assert cur.fetchone() == (0,)

            # Idempotencja: ponowny przebieg nie nadpisuje waluty ustawionej importem.
            cur.execute("UPDATE positions SET currency = 'USD' WHERE symbol = 'VUSA.L'")
            cur.execute(migration)
            assert self._currencies(cur)["VUSA.L"] == "USD"


@pytest.mark.containers
class TestMigration037BackfillsExperimentArms:
    """EXP-05: ramiona sprzed migracji żyją tylko w kwicie v2
    (`decision_receipts->'experiment_arms'`); bez kwitu zostaje NULL."""

    def _apply_up_to_036(self, conn: psycopg2.extensions.connection) -> None:
        idx = MIGRATION_FILES.index("037_experiment_arms.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())

    def test_037_copies_the_arms_from_the_v2_receipt(self, fresh_pg) -> None:
        conn = fresh_pg
        self._apply_up_to_036(conn)
        migration = (MIGRATIONS_DIR / "037_experiment_arms.sql").read_text()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction_logs "
                "(symbol, timestamp, price_at_prediction, decision_receipts) VALUES "
                "('EB_ARMS', '2026-11-02T21:30:00+00:00', 100.0, "
                "'{\"schema_version\": 2, \"experiment_arms\": {\"EXP-03\": \"no_memory\"}}'), "
                "('EB_NONE', '2026-11-02T21:30:00+00:00', 100.0, '{\"schema_version\": 2}'), "
                "('EB_BARE', '2026-11-02T21:30:00+00:00', 100.0, NULL)"
            )

            cur.execute(migration)
            cur.execute(
                "SELECT symbol, experiment_arms FROM prediction_logs WHERE symbol LIKE 'EB_%'"
            )
            arms = dict(cur.fetchall())

            assert arms == {
                "EB_ARMS": {"EXP-03": "no_memory"},
                "EB_NONE": None,
                "EB_BARE": None,
            }
            # Idempotencja: ponowny przebieg nie nadpisuje ramion zapisanych przez Fast Loop.
            cur.execute(
                "UPDATE prediction_logs SET experiment_arms = '{\"EXP-03\": \"memory\"}' "
                "WHERE symbol = 'EB_ARMS'"
            )
            cur.execute(migration)
            cur.execute("SELECT experiment_arms FROM prediction_logs WHERE symbol = 'EB_ARMS'")
            assert cur.fetchone() == ({"EXP-03": "memory"},)


@pytest.mark.containers
class TestMigration045PersonaVoteOutcomes:
    """WIA-51: RPC 045 zwraca SUROWE rozliczone głosy person, a trafność liczy
    domena — SQL z 015/018 oceniał HOLD na krypto pasmem akcji ±0.5%.
    WIA-46: oś czasu `predicted_at` / `settled_at` pozwala policzyć wagi, które
    Fast Loop znał w chwili predykcji."""

    @staticmethod
    def _settled_vote(
        cur: psycopg2.extensions.cursor,
        symbol: str,
        investor: str,
        recommendation: str,
        actual: float,
        *,
        age: str,
        settled_age: str | None,
    ) -> None:
        # INSERT z jawnym settled_at — trigger 029 stempluje tylko UPDATE.
        cur.execute(
            "INSERT INTO prediction_logs "
            "(symbol, timestamp, price_at_prediction, actual_price_after_12h, settled_at) "
            "VALUES (%s, NOW() - %s::interval, 100.0, %s, NOW() - %s::interval) "
            "RETURNING id",
            (symbol, age, actual, settled_age),
        )
        prediction_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO council_votes "
            "(prediction_id, symbol, investor_name, recommendation, confidence) "
            "VALUES (%s, %s, %s, %s, 0.7)",
            (prediction_id, symbol, investor, recommendation),
        )

    @staticmethod
    def _outcomes(cur: psycopg2.extensions.cursor) -> list[PersonaVoteOutcome]:
        cur.execute(
            "SELECT prediction_id, investor_name, recommendation, confidence, symbol, "
            "price_at_prediction, actual_price_after_12h, predicted_at, settled_at "
            "FROM persona_vote_outcomes(%s)",
            (90,),
        )
        return [
            PersonaVoteOutcome(
                prediction_id=str(pid),
                investor_name=name,
                recommendation=rec,
                confidence=conf,
                symbol=symbol,
                price_at_prediction=Decimal(str(price)),
                actual_price=Decimal(str(actual)),
                predicted_at=predicted_at,
                settled_at=settled_at,
            )
            for pid, name, rec, conf, symbol, price, actual, predicted_at, settled_at
            in cur.fetchall()
        ]

    def test_persona_vote_outcomes_return_raw_settled_votes(self, pg_conn) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            # Izolacja: RPC czyta całą tabelę, a kontener dzielą testy modułu.
            cur.execute("DELETE FROM council_votes")
            self._settled_vote(
                cur, "PVO_BTC", "Buffett", "HOLD", 102.0, age="3 days", settled_age="2 days"
            )
            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction) "
                "VALUES ('PVO_OPEN', 100.0) RETURNING id"
            )
            open_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO council_votes "
                "(prediction_id, symbol, investor_name, recommendation, confidence) "
                "VALUES (%s, 'PVO_OPEN', 'Buffett', 'BUY', 0.7)",
                (open_id,),
            )
            outcomes = self._outcomes(cur)

        (outcome,) = outcomes
        assert (outcome.investor_name, outcome.recommendation, outcome.symbol) == (
            "Buffett", "HOLD", "PVO_BTC"
        )
        assert outcome.actual_price == Decimal("102.0000")
        assert outcome.settled_at is not None and outcome.predicted_at is not None
        assert outcome.settled_at > outcome.predicted_at
        # Ta sama liczba, dwie oceny: domena z pasmem krypto, SQL 018 bez niego.
        assert persona_hit_stats(outcomes, crypto_symbols={"PVO_BTC"}) == {"Buffett": (1.0, 1)}

    def test_persona_weights_as_of_ignore_votes_settled_after_cutoff(self, pg_conn) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM council_votes")
            cur.execute("SELECT NOW() - interval '2 days'")
            cutoff = cur.fetchone()[0]
            self._settled_vote(
                cur, "PVO_EARLY", "Wood", "BUY", 110.0, age="5 days", settled_age="4 days"
            )
            self._settled_vote(
                cur, "PVO_LATE", "Burry", "BUY", 110.0, age="3 days", settled_age="1 day"
            )
            outcomes = self._outcomes(cur)

        weights = persona_accuracy_weights_as_of(outcomes, as_of=cutoff, window_days=90)

        assert weights == {"Wood": pytest.approx(1.5)}

    def test_045_is_idempotent_and_keeps_the_old_rpcs(self, pg_conn) -> None:
        migration = (MIGRATIONS_DIR / "045_persona_vote_outcomes.sql").read_text()
        with pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute("SELECT count(*) FROM persona_accuracy_weights(90)")
            cur.execute("SELECT count(*) FROM persona_accuracy_stats(90)")
            cur.execute(
                "SELECT obj_description('persona_vote_outcomes(integer)'::regprocedure)"
            )
            assert "WIA-51" in cur.fetchone()[0]


@pytest.mark.containers
class TestMigration065PersonaScoring:
    """WIA-44: RPC `persona_scoring_stats` — hit-rate, liczba głosów, średni Brier
    (confidence − trafienie)² i log loss deklarowanej pewności głosu, tą samą regułą
    trafności co domena (pasmo HOLD krypto z parametru, bez rozliczeń `stale_quote`)."""

    @staticmethod
    def _settled_vote(
        cur: psycopg2.extensions.cursor,
        symbol: str,
        investor: str,
        recommendation: str,
        confidence: float,
        actual: float,
        *,
        stale: bool | None = None,
    ) -> None:
        cur.execute(
            "INSERT INTO prediction_logs "
            "(symbol, timestamp, price_at_prediction, actual_price_after_12h, stale_quote) "
            "VALUES (%s, NOW() - interval '3 days', 100.0, %s, %s) RETURNING id",
            (symbol, actual, stale),
        )
        prediction_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO council_votes "
            "(prediction_id, symbol, investor_name, recommendation, confidence) "
            "VALUES (%s, %s, %s, %s, %s)",
            (prediction_id, symbol, investor, recommendation, confidence),
        )

    @staticmethod
    def _scoring(
        cur: psycopg2.extensions.cursor, crypto: list[str]
    ) -> dict[str, tuple[float, int, float, float]]:
        cur.execute(
            "SELECT investor_name, hit_rate, vote_count, brier, log_loss "
            "FROM persona_scoring_stats(90, %s)",
            (crypto,),
        )
        return {name: (hit, count, brier, loss) for name, hit, count, brier, loss in cur.fetchall()}

    def test_persona_scoring_rpc_returns_expected_brier(self, pg_conn) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM council_votes")
            # Przepewna: BUY 0.9 trafia, BUY 0.9 pudłuje → hit-rate 0.5, Brier (0.01 + 0.81) / 2.
            self._settled_vote(cur, "PS65_A", "Over", "BUY", 0.9, 110.0)
            self._settled_vote(cur, "PS65_B", "Over", "BUY", 0.9, 90.0)
            # Skalibrowana: BUY 0.5 trafia, BUY 0.5 pudłuje → Brier 0.25.
            self._settled_vote(cur, "PS65_C", "Calibrated", "BUY", 0.5, 110.0)
            self._settled_vote(cur, "PS65_D", "Calibrated", "BUY", 0.5, 90.0)
            scoring = self._scoring(cur, [])

        assert scoring["Over"][:2] == (0.5, 2)
        assert scoring["Over"][2] == pytest.approx((0.01 + 0.81) / 2)
        assert scoring["Over"][3] == pytest.approx(-(math.log(0.9) + math.log(0.1)) / 2)
        assert scoring["Calibrated"][2] == pytest.approx(0.25)
        assert scoring["Calibrated"][3] == pytest.approx(math.log(2))

    def test_scoring_matches_the_domain_rule_including_the_crypto_hold_band(
        self, pg_conn
    ) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM council_votes")
            # HOLD +2% mieści się w paśmie krypto ±2.5%, a nie w paśmie akcji ±0.5%.
            self._settled_vote(cur, "PS65_BTC", "Hold", "HOLD", 0.8, 102.0)
            self._settled_vote(cur, "PS65_AAPL", "Hold", "HOLD", 0.8, 102.0)
            self._settled_vote(cur, "PS65_SELL", "Hold", "SELL", 0.7, 95.0)
            cur.execute(
                "SELECT prediction_id, investor_name, recommendation, confidence, symbol, "
                "price_at_prediction, actual_price_after_12h, predicted_at, settled_at "
                "FROM persona_vote_outcomes(90)"
            )
            outcomes = [
                PersonaVoteOutcome(
                    prediction_id=str(pid), investor_name=name, recommendation=rec,
                    confidence=conf, symbol=symbol,
                    price_at_prediction=Decimal(str(price)), actual_price=Decimal(str(actual)),
                    predicted_at=predicted_at, settled_at=settled_at,
                )
                for pid, name, rec, conf, symbol, price, actual, predicted_at, settled_at
                in cur.fetchall()
            ]
            scoring = self._scoring(cur, ["PS65_BTC"])

        crypto = {"PS65_BTC"}
        samples = [
            (o.confidence, persona_vote_correct(
                o.recommendation, o.price_at_prediction, o.actual_price,
                AssetType.CRYPTO if o.symbol in crypto else None,
            ))
            for o in outcomes
        ]
        hit_rate, votes = persona_hit_stats(outcomes, crypto_symbols=crypto)["Hold"]
        assert scoring["Hold"][0] == pytest.approx(hit_rate)
        assert scoring["Hold"][1] == votes == 3
        assert scoring["Hold"][2] == pytest.approx(brier_score(samples))
        assert scoring["Hold"][3] == pytest.approx(log_loss(samples))

    def test_settlements_on_a_stale_quote_are_not_scored(self, pg_conn) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM council_votes")
            self._settled_vote(cur, "PS65_FRESH", "Wood", "BUY", 0.6, 110.0, stale=False)
            self._settled_vote(cur, "PS65_UNKNOWN", "Wood", "BUY", 0.6, 110.0, stale=None)
            self._settled_vote(cur, "PS65_STALE", "Wood", "BUY", 0.6, 100.0, stale=True)
            scoring = self._scoring(cur, [])

        assert scoring["Wood"][1] == 2

    def test_open_predictions_are_not_scored(self, pg_conn) -> None:
        _apply_migrations(pg_conn)
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM council_votes")
            cur.execute(
                "INSERT INTO prediction_logs (symbol, price_at_prediction) "
                "VALUES ('PS65_OPEN', 100.0) RETURNING id"
            )
            open_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO council_votes "
                "(prediction_id, symbol, investor_name, recommendation, confidence) "
                "VALUES (%s, 'PS65_OPEN', 'Wood', 'BUY', 0.6)",
                (open_id,),
            )
            assert self._scoring(cur, []) == {}

    def test_065_is_idempotent_and_keeps_the_old_rpcs(self, pg_conn) -> None:
        migration = (MIGRATIONS_DIR / "065_persona_scoring.sql").read_text()
        with pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute("SELECT count(*) FROM persona_accuracy_stats(90)")
            cur.execute("SELECT count(*) FROM persona_vote_outcomes(90)")
            cur.execute(
                "SELECT obj_description('persona_scoring_stats(integer, text[])'::regprocedure)"
            )
            assert "WIA-44" in cur.fetchone()[0]


@pytest.mark.containers
class TestMigration046CorrectionCategory:
    """WIA-57: kategoria przyczyny błędu z diagnozy — kolumna z CHECK zbioru
    `MistakeCategory`; NULL dla wierszy bez kategorii."""

    def test_correction_category_accepts_every_domain_category_and_null(
        self, pg_conn
    ) -> None:
        from src.domain.mistake_taxonomy import MistakeCategory

        _apply_migrations(pg_conn)
        assert "correction_category" in _columns(pg_conn, "prediction_logs")
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM prediction_logs WHERE symbol LIKE 'CC046_%'")
            for index, category in enumerate([*MistakeCategory, None]):
                cur.execute(
                    "INSERT INTO prediction_logs (symbol, timestamp, correction_category) "
                    "VALUES (%s, NOW() - %s * interval '1 hour', %s)",
                    (f"CC046_{index}", index, None if category is None else category.value),
                )
            cur.execute(
                "SELECT count(*) FROM prediction_logs WHERE symbol LIKE 'CC046_%'"
            )
            assert cur.fetchone() == (len(MistakeCategory) + 1,)

    def test_correction_category_rejects_a_value_outside_the_taxonomy(self, pg_conn) -> None:
        with pg_conn.cursor() as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO prediction_logs (symbol, correction_category) "
                "VALUES ('CC046_BAD', 'ASTROLOGY')"
            )

    def test_046_is_idempotent(self, pg_conn) -> None:
        migration = (MIGRATIONS_DIR / "046_correction_category.sql").read_text()
        with pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT col_description('prediction_logs'::regclass, attnum) "
                "FROM pg_attribute WHERE attrelid = 'prediction_logs'::regclass "
                "AND attname = 'correction_category'"
            )
            assert "WIA-57" in cur.fetchone()[0]


@pytest.mark.containers
class TestMigration048NormalizesPredictedTrend:
    """ML-19: UPDATE do UPPER(TRIM) idzie PRZED CHECK, a CHECK jest NOT VALID —
    inaczej migracja padłaby na pierwszym historycznym wierszu spoza listy."""

    def test_048_normalizes_existing_trends_before_the_check(self, fresh_pg) -> None:
        conn = fresh_pg
        migration = (MIGRATIONS_DIR / "048_normalize_predicted_trend.sql").read_text()
        idx = MIGRATION_FILES.index("048_normalize_predicted_trend.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, predicted_trend) VALUES "
                "('NT_SPACED', '2026-06-01T10:00:00+00:00', ' bullish '), "
                "('NT_MIXED', '2026-06-01T10:00:00+00:00', 'Bearish'), "
                "('NT_LEGACY', '2026-06-01T10:00:00+00:00', 'UP'), "
                "('NT_NULL', '2026-06-01T10:00:00+00:00', NULL)"
            )

            # Wiersz spoza listy w historii nie wywala migracji (NOT VALID).
            cur.execute(migration)
            # Idempotencja: powtórka nie dubluje ograniczenia ani nie pada.
            cur.execute(migration)

            cur.execute(
                "SELECT symbol, predicted_trend FROM prediction_logs WHERE symbol LIKE 'NT_%'"
            )
            trends = dict(cur.fetchall())
            cur.execute(
                "SELECT convalidated FROM pg_constraint "
                "WHERE conname = 'prediction_logs_predicted_trend_known'"
            )
            validated = [row[0] for row in cur.fetchall()]

            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO prediction_logs (symbol, predicted_trend) VALUES ('NT_NEW', 'UP')"
                )

        assert trends == {
            "NT_SPACED": "BULLISH",
            "NT_MIXED": "BEARISH",
            "NT_LEGACY": "UP",
            "NT_NULL": None,
        }
        assert validated == [False]

@pytest.mark.containers
class TestMigration053EarningsCache:
    """FIN-11: `earnings_cache` — jedna data raportu na spółkę, odświeżana upsertem
    w Slow Loopie, zamknięta dla ról publicznych jak każda tabela po 038."""

    def test_earnings_cache_holds_one_report_date_per_symbol(self, pg_conn) -> None:
        _apply_migrations(pg_conn)

        assert _columns(pg_conn, "earnings_cache") == {"symbol", "report_date", "fetched_at"}
        assert "idx_earnings_cache_fetched_at" in _indexes(pg_conn, "earnings_cache")
        with pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO earnings_cache (symbol, report_date) VALUES ('EC_AAPL', '2026-10-29')"
            )
            cur.execute(
                "INSERT INTO earnings_cache (symbol, report_date, fetched_at) "
                "VALUES ('EC_AAPL', '2026-10-30', '2026-11-01T03:00:00+00:00') "
                "ON CONFLICT (symbol) DO UPDATE SET report_date = EXCLUDED.report_date, "
                "fetched_at = EXCLUDED.fetched_at"
            )
            cur.execute(
                "SELECT count(*), max(report_date)::text FROM earnings_cache "
                "WHERE symbol = 'EC_AAPL'"
            )
            assert cur.fetchone() == (1, "2026-10-30")
            with pytest.raises(psycopg2.errors.NotNullViolation):
                cur.execute("INSERT INTO earnings_cache (symbol) VALUES ('EC_NONE')")

    def test_earnings_cache_is_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'earnings_cache'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [
                _has_table_privilege(cur, role, "earnings_cache") for role in PUBLIC_API_ROLES
            ]

            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO earnings_cache (symbol, report_date) VALUES ('EC_SR', '2026-10-29')"
            )
            cur.execute("SELECT report_date::text FROM earnings_cache WHERE symbol = 'EC_SR'")
            assert cur.fetchone() == ("2026-10-29",)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM earnings_cache LIMIT 1")


@pytest.mark.containers
class TestMigration054CouncilVotesUnique:
    """NIEZ-14: re-run Fast Loopa w tej samej godzinie dublował głosy rady (goły
    insert), a każdy duplikat ważył w rankingu person. 054 najpierw usuwa duplikaty
    (zostaje najwyższe `id`, czyli najnowszy głos), potem zakłada UNIQUE."""

    @staticmethod
    def _prediction(cur: psycopg2.extensions.cursor, symbol: str) -> str:
        cur.execute(
            "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) "
            "VALUES (%s, '2026-10-20T21:30:00+00:00', 100.0) RETURNING id",
            (symbol,),
        )
        return str(cur.fetchone()[0])

    @staticmethod
    def _vote(cur: psycopg2.extensions.cursor, prediction_id: str, name: str, rec: str) -> None:
        cur.execute(
            "INSERT INTO council_votes "
            "(prediction_id, symbol, investor_name, recommendation, confidence) "
            "VALUES (%s, 'CV', %s, %s, 0.7)",
            (prediction_id, name, rec),
        )

    def test_council_votes_rejects_duplicate_vote(self, pg_conn) -> None:
        _apply_migrations(pg_conn)

        with pg_conn.cursor() as cur:
            prediction_id = self._prediction(cur, "CV_UNIQ")
            self._vote(cur, prediction_id, "Warren Buffett", "BUY")
            with pytest.raises(psycopg2.errors.UniqueViolation):
                self._vote(cur, prediction_id, "Warren Buffett", "SELL")
            cur.execute(
                "INSERT INTO council_votes "
                "(prediction_id, symbol, investor_name, recommendation, confidence) "
                "VALUES (%s, 'CV', 'Warren Buffett', 'HOLD', 0.5) "
                "ON CONFLICT (prediction_id, investor_name) DO UPDATE "
                "SET recommendation = EXCLUDED.recommendation",
                (prediction_id,),
            )
            cur.execute(
                "SELECT count(*), max(recommendation) FROM council_votes "
                "WHERE prediction_id = %s",
                (prediction_id,),
            )
            assert cur.fetchone() == (1, "HOLD")

    def test_054_dedupes_then_builds_the_unique_index(self, fresh_pg) -> None:
        conn = fresh_pg
        idx = MIGRATION_FILES.index("054_council_votes_unique.sql")
        with conn.cursor() as cur:
            for filename in MIGRATION_FILES[:idx]:
                cur.execute((MIGRATIONS_DIR / filename).read_text())
            prediction_id = self._prediction(cur, "CV_DUP")
            self._vote(cur, prediction_id, "Warren Buffett", "BUY")
            self._vote(cur, prediction_id, "Warren Buffett", "SELL")
            self._vote(cur, prediction_id, "Michael Burry", "SELL")

            migration = (MIGRATIONS_DIR / "054_council_votes_unique.sql").read_text()
            cur.execute(migration)
            cur.execute(migration)

            cur.execute(
                "SELECT investor_name, recommendation FROM council_votes "
                "WHERE prediction_id = %s ORDER BY investor_name",
                (prediction_id,),
            )
            assert cur.fetchall() == [("Michael Burry", "SELL"), ("Warren Buffett", "SELL")]
        assert "idx_council_votes_prediction_investor_uniq" in _indexes(conn, "council_votes")


@pytest.mark.containers
class TestMigration058SettledReadsWithoutStaleQuote:
    """DAT-14: rozliczenie z notowania sprzed sesji po predykcji (`stale_quote` =
    true, 055) ma cenę równą cenie wejścia — 058 wycina je z danych treningowych
    i z trafności person (RPC 015, 018 i 045). `IS NOT TRUE`: NULL (krypto, wiersze
    sprzed 055) zostaje."""

    @staticmethod
    def _settled(cur: psycopg2.extensions.cursor, symbol: str, stale: bool | None) -> str:
        cur.execute(
            "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction, "
            "actual_price_after_12h, stale_quote) "
            "VALUES (%s, NOW() - interval '2 days', 100.0, 102.0, %s) RETURNING id",
            (symbol, stale),
        )
        return str(cur.fetchone()[0])

    @staticmethod
    def _vote(cur: psycopg2.extensions.cursor, prediction_id: str, symbol: str) -> None:
        cur.execute(
            "INSERT INTO council_votes "
            "(prediction_id, symbol, investor_name, recommendation, confidence) "
            "VALUES (%s, %s, 'SQ058 Persona', 'BUY', 0.7)",
            (prediction_id, symbol),
        )

    def test_feature_store_drops_stale_settlements(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        with _rolled_back(migrated_pg_conn) as cur:
            for symbol, stale in (("SQ058_T", True), ("SQ058_F", False), ("SQ058_N", None)):
                self._settled(cur, symbol, stale)
            cur.execute("REFRESH MATERIALIZED VIEW ml_feature_store")
            cur.execute(
                "SELECT symbol FROM ml_feature_store WHERE symbol LIKE 'SQ058_%' ORDER BY symbol"
            )
            symbols = [row[0] for row in cur.fetchall()]

        assert symbols == ["SQ058_F", "SQ058_N"]

    def test_feature_store_after_058_keeps_the_lag_columns_and_indexes(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        """058 odtwarza widok z 050 1:1 — nowy jest tylko filtr."""
        assert {"settled_at", "label_horizon_hours", "price_prev_12h"} <= _columns(
            migrated_pg_conn, "ml_feature_store"
        )
        assert {
            "idx_ml_feature_store_id",
            "idx_ml_feature_store_symbol_timestamp",
        } <= _indexes(migrated_pg_conn, "ml_feature_store")
        with migrated_pg_conn.cursor() as cur:
            cur.execute("SELECT refresh_ml_feature_store()")
            cur.execute("SELECT obj_description('ml_feature_store'::regclass)")
            assert "058" in cur.fetchone()[0]

    def test_persona_reads_drop_stale_settlements(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        with _rolled_back(migrated_pg_conn) as cur:
            self._vote(cur, self._settled(cur, "SQ058_VT", True), "SQ058_VT")
            self._vote(cur, self._settled(cur, "SQ058_VN", None), "SQ058_VN")
            cur.execute(
                "SELECT symbol FROM persona_vote_outcomes(90) "
                "WHERE investor_name = 'SQ058 Persona'"
            )
            outcomes = [row[0] for row in cur.fetchall()]
            cur.execute(
                "SELECT vote_count FROM persona_accuracy_stats(90) "
                "WHERE investor_name = 'SQ058 Persona'"
            )
            stats = cur.fetchall()
            cur.execute(
                "SELECT weight FROM persona_accuracy_weights(90) "
                "WHERE investor_name = 'SQ058 Persona'"
            )
            weights = cur.fetchall()

        assert outcomes == ["SQ058_VN"]
        assert stats == [(1,)]
        # Jeden trafiony BUY (102 > 100): waga 0.5 + 1.0 = 1.5 — bez stęchłego głosu.
        assert weights == [(pytest.approx(1.5),)]

    def test_058_is_idempotent(self, migrated_pg_conn: psycopg2.extensions.connection) -> None:
        migration = (MIGRATIONS_DIR / "058_settled_reads_without_stale_quote.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT obj_description('persona_vote_outcomes(integer)'::regprocedure)"
            )
            assert "DAT-14" in cur.fetchone()[0]


@pytest.mark.containers
class TestMigration059SettlementSessionClose:
    """WIA-61: druga cena rozliczenia — oficjalne zamknięcie sesji z datą, źródłem
    i trafnością obok `actual_price_after_12h`; NULL dla krypto i starych wierszy."""

    def test_settlement_carries_the_session_close_next_to_the_quote(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert {
            "settlement_close_price",
            "settlement_session_date",
            "settlement_source",
            "is_trend_correct_session_close",
        } <= _columns(migrated_pg_conn, "prediction_logs")
        with _rolled_back(migrated_pg_conn) as cur:
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, price_at_prediction) VALUES "
                "('SC059', '2026-10-19T05:30:00+00:00', 100.0) RETURNING id"
            )
            prediction_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE prediction_logs SET actual_price_after_12h = 105.0, "
                "settlement_close_price = 106.12345, settlement_session_date = '2026-10-19', "
                "settlement_source = 'quote_close', is_trend_correct_session_close = true "
                "WHERE id = %s RETURNING actual_price_after_12h, settlement_close_price, "
                "settlement_session_date::text, settlement_source, is_trend_correct_session_close",
                (prediction_id,),
            )
            row = cur.fetchone()

        assert row == (
            Decimal("105.0000"), Decimal("106.1235"), "2026-10-19", "quote_close", True
        )

    def test_settlement_source_rejects_an_unknown_field(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        with (
            _rolled_back(migrated_pg_conn) as cur,
            pytest.raises(psycopg2.errors.CheckViolation),
        ):
            cur.execute(
                "INSERT INTO prediction_logs (symbol, timestamp, settlement_source) "
                "VALUES ('SC059_BAD', '2026-10-19T05:30:00+00:00', 'vendor_close')"
            )

    def test_059_is_idempotent(self, migrated_pg_conn: psycopg2.extensions.connection) -> None:
        migration = (MIGRATIONS_DIR / "059_settlement_session_close.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT count(*) FROM pg_constraint "
                "WHERE conrelid = 'prediction_logs'::regclass AND contype = 'c' "
                "AND pg_get_constraintdef(oid) LIKE '%settlement_source%'"
            )
            assert cur.fetchone() == (1,)


@pytest.mark.containers
class TestMigration062CorporateActions:
    """DAT-23: podejrzenia splitów — jeden wiersz na symbol, rodzaj i dzień UTC,
    zamknięte dla ról publicznych jak każda tabela po 038."""

    _INSERT = (
        "INSERT INTO corporate_actions "
        "(symbol, kind, ratio, prev_close, reference_price, detected_at, detected_on) "
        "VALUES (%s, 'split_suspected', %s, 120.6, 1200, '2026-10-27T05:30:00+00:00', "
        "'2026-10-27') ON CONFLICT (symbol, kind, detected_on) DO NOTHING"
    )

    def test_split_suspicion_is_stored_once_per_symbol_kind_and_day(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert _columns(migrated_pg_conn, "corporate_actions") == {
            "id", "symbol", "kind", "ratio", "prev_close", "reference_price",
            "detected_at", "detected_on",
        }
        with _rolled_back(migrated_pg_conn) as cur:
            cur.execute(self._INSERT, ("CA_NVDA", Decimal("0.1")))
            cur.execute(self._INSERT, ("CA_NVDA", Decimal("0.1")))
            cur.execute(
                "SELECT count(*), max(ratio) FROM corporate_actions WHERE symbol = 'CA_NVDA'"
            )
            assert cur.fetchone() == (1, Decimal("0.1"))

    @pytest.mark.parametrize(
        ("kind", "ratio"), [("dividend", Decimal("0.1")), ("split_suspected", Decimal("0"))]
    )
    def test_unknown_kind_and_non_positive_ratio_are_rejected(
        self, migrated_pg_conn: psycopg2.extensions.connection, kind: str, ratio: Decimal
    ) -> None:
        with _rolled_back(migrated_pg_conn) as cur, pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO corporate_actions "
                "(symbol, kind, ratio, prev_close, reference_price, detected_at, detected_on) "
                "VALUES ('CA_BAD', %s, %s, 1, 1, now(), current_date)",
                (kind, ratio),
            )

    def test_corporate_actions_are_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute(
                "SELECT relrowsecurity FROM pg_class WHERE oid = 'corporate_actions'::regclass"
            )
            assert cur.fetchone() == (True,)
            public = [
                _has_table_privilege(cur, role, "corporate_actions") for role in PUBLIC_API_ROLES
            ]

            _set_role(cur, "service_role")
            cur.execute(self._INSERT, ("CA_SR", Decimal("2")))
            cur.execute("SELECT ratio FROM corporate_actions WHERE symbol = 'CA_SR'")
            assert cur.fetchone() == (Decimal("2"),)

        assert public == [False, False]
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cur.execute("SELECT 1 FROM corporate_actions LIMIT 1")

    def test_062_is_idempotent(self, migrated_pg_conn: psycopg2.extensions.connection) -> None:
        migration = (MIGRATIONS_DIR / "062_corporate_actions.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT count(*) FROM pg_constraint "
                "WHERE conrelid = 'corporate_actions'::regclass AND contype = 'u'"
            )
            assert cur.fetchone() == (1,)


@pytest.mark.containers
class TestMigration063RagExactSearch:
    """ML-23: `match_news_embeddings_exact` — kontrakt `match_news_embeddings`, ale
    dokładne KNN. ivfflat z lists = 100 przy probes = 1 przeszukuje ~1% list, więc
    najbliższy analog bywa pominięty, a z filtrem reżimu brakuje wierszy (sprawdzone
    2026-10-25 na tym samym zbiorze: stara funkcja oddała 2 z 5 wierszy reżimu)."""

    _ROWS = 500
    _DIMENSIONS = 1536

    @classmethod
    def _vectors(cls) -> list[str]:
        import random

        rng = random.Random(23)
        return [
            "[" + ",".join(f"{rng.uniform(-1, 1):.4f}" for _ in range(cls._DIMENSIONS)) + "]"
            for _ in range(cls._ROWS)
        ]

    def _insert(self, cur: psycopg2.extensions.cursor, vectors: list[str]) -> None:
        """Wektory testu jako jedyne embeddingi tabeli (transakcja jest wycofywana) i plan
        jak na produkcji: indeks ivfflat przebudowany na danych (prawdziwe centroidy),
        a planer bez skanu sekwencyjnego — na 500 wierszach sam wybrałby dokładny skan."""
        cur.execute("DELETE FROM prediction_logs WHERE embedding IS NOT NULL")
        for index, vector in enumerate(vectors):
            cur.execute(
                "INSERT INTO prediction_logs (symbol, news_summary, regime, embedding) "
                "VALUES (%s, 'news', %s, %s::vector)",
                (f"RAG{index:03d}", "RISK_OFF" if index % 2 else "RISK_ON", vector),
            )
        cur.execute("REINDEX INDEX idx_prediction_logs_embedding")
        cur.execute("SET LOCAL enable_seqscan = off")

    def test_match_news_embeddings_returns_exact_nearest_neighbour(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        vectors = self._vectors()
        with _rolled_back(migrated_pg_conn) as cur:
            self._insert(cur, vectors)
            cur.execute(
                "SELECT symbol FROM match_news_embeddings_exact(%s::vector, 1)", (vectors[137],)
            )
            top = cur.fetchall()
            cur.execute(
                "SELECT count(*) FROM match_news_embeddings_exact(%s::vector, 5, 'RISK_OFF')",
                (vectors[0],),
            )
            filtered = cur.fetchone()

        assert top == [("RAG137",)]
        assert filtered == (5,)

    def test_service_role_executes_the_exact_search(self, supabase_like_conn) -> None:
        vector = "[" + ",".join(["0.1"] * 1536) + "]"
        with _rolled_back(supabase_like_conn) as cur:
            _set_role(cur, "service_role")
            cur.execute(
                "SELECT count(*) FROM match_news_embeddings_exact(%s::vector, 3)", (vector,)
            )
            assert cur.fetchone() == (0,)

    def test_063_is_idempotent_and_keeps_the_approximate_search(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        migration = (MIGRATIONS_DIR / "063_rag_exact_search.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT proname FROM pg_proc "
                "WHERE proname LIKE 'match_news_embeddings%%' ORDER BY proname"
            )
            names = [row[0] for row in cur.fetchall()]

        assert names == ["match_news_embeddings", "match_news_embeddings_exact"]
        assert "idx_prediction_logs_embedding" in _indexes(migrated_pg_conn, "prediction_logs")


def _full_analysis_record() -> dict[str, object]:
    """TEST-14: rekord `prediction_logs` z PEŁNEJ analizy grafu przy każdej fladze
    i porcie, który dokłada klucz (kwit v2, przepis, atestacja, opcje, przedział,
    rozkład trendu, notowanie, embedding), PO serializacji adaptera — dokładnie ten
    słownik idzie do PostgREST."""
    from datetime import timedelta
    from unittest.mock import MagicMock, Mock, patch

    from src.application.agent_graph import AgentGraphDeps, create_agent_graph
    from src.application.ports import (
        AttestationPublisherPort,
        EmbeddingPort,
        LLMPort,
        MarketDataPort,
        MLPredictionPort,
        NewsPort,
        OptionsFlowPort,
        QuotePort,
        RepositoryPort,
        SentimentPort,
    )
    from src.domain.asset import Asset
    from src.domain.conformal import CalibrationPoint
    from src.domain.options_flow import OptionsFlowSnapshot
    from src.domain.recipe import PredictionRecipe
    from src.domain.value_objects import AssetType, Money, Quote, Threshold
    from src.infrastructure.adapters.supabase_repo import SupabaseRepository

    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    repository = Mock(spec=RepositoryPort)
    repository.get_unverified_prediction.return_value = None
    repository.get_last_prediction_price.return_value = Money(Decimal("100"))
    repository.find_similar_predictions.return_value = []
    repository.get_current_hour_commitment.return_value = None
    repository.save_prediction.return_value = "pred-1"
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.3,
        "av_relevance_avg": 0.7,
        "news_volume_24h": 4,
        "high_relevance_count": 2,
        "av_sentiment_label": "Bullish",
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Reuters: Apple beats", "source": "Reuters"}]
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH",
        "confidence_score": 0.8,
        "av_agreement": 0.7,
        "target_price_12h": 110.0,
        "reasoning": "Momentum.",
        "probabilities": {"BULLISH": 0.6, "BEARISH": 0.1, "SIDEWAYS": 0.3},
    }
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("108"))
    ml.explain.return_value = {"price_delta": 0.01}
    embedding = Mock(spec=EmbeddingPort)
    embedding.embed.return_value = [0.01] * 1536
    options = Mock(spec=OptionsFlowPort)
    options.get_options_flow.return_value = OptionsFlowSnapshot("AAPL", 0.6, implied_vol=0.4)
    quotes = Mock(spec=QuotePort)
    quotes.get_quote.return_value = Quote(
        price=Money(Decimal("106")),
        quoted_at=now - timedelta(hours=64),
        prev_close=Money(Decimal("104")),
    )
    window = [CalibrationPoint(0.001 * i, 0.002 * i - 0.01) for i in range(30)]
    deps = AgentGraphDeps(
        market_port=Mock(spec=MarketDataPort),
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repository,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        embedding_port=embedding,
        options_port=options,
        quote_port=quotes,
        attestation_publisher=Mock(spec=AttestationPublisherPort),
        vector_memory_enabled=True,
        regime_tagging_enabled=True,
        receipts_enabled=True,
        feature_set_version_enabled=True,
        recipe=PredictionRecipe(
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-6",
            council_provider="openai",
            council_model="gpt-5-mini",
            prompt_sha256="a" * 64,
            personas_sha256="b" * 64,
            config_hash="c" * 64,
            model_version="cold_start",
            universe_version="d" * 64,
            git_sha="local",
            run_id="local",
        ),
        recipe_version_enabled=True,
        move_calibration={AssetType.STOCK: window},
        interval_columns_enabled=True,
        gate_delta_enabled=True,
        experiment_arms_enabled=True,
        trend_probabilities_enabled=True,
        stale_quote_enabled=True,
        clock=lambda: now,
    )
    create_agent_graph(deps).compile().invoke(
        {
            "symbol": "AAPL",
            "previous_price": Decimal("100"),
            "asset": Asset(symbol="AAPL", asset_type=AssetType.STOCK),
            "regime_context": "NEUTRAL",
            "experiment_arms": {"EXP-02": "blind"},
        }
    )
    record = repository.save_prediction.call_args.args[0]

    client = MagicMock()
    client.table.return_value.upsert.return_value.execute.return_value.data = [{"id": "x"}]
    with patch("src.infrastructure.adapters.supabase_repo.create_client", return_value=client):
        SupabaseRepository(url="https://test.supabase.co", key="key").save_prediction(record)
    serialized: dict[str, object] = client.table.return_value.upsert.call_args.args[0]
    return serialized


@pytest.mark.containers
class TestSaveNodeRecordFitsPredictionLogs:
    """TEST-14 (G69): PGRST204 to powracająca awaria produkcyjna — klucz rekordu
    bez kolumny zabija zapis całej, już opłaconej predykcji. Rekord pełnej analizy
    po serializacji musi mieścić się w kolumnach `prediction_logs` po migracjach."""

    # Klucze, które dokładają flagi i porty — pilnuje, że test je naprawdę wywołał.
    _OPTIONAL_KEYS = frozenset({
        "regime",
        "decision_receipts",
        "feature_set_version",
        "recipe_version",
        "edge_sigma",
        "interval_low",
        "interval_high",
        "gate_delta",
        "experiment_arms",
        "trend_probabilities",
        "stale_quote",
        "commitment_hash",
        "commitment_salt",
        "commitment_payload",
        "embedding",
    })

    def test_saved_record_keys_are_prediction_logs_columns(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        record = _full_analysis_record()

        assert set(record) >= self._OPTIONAL_KEYS, (
            f"graf nie zapisał kluczy: {sorted(self._OPTIONAL_KEYS - set(record))}"
        )
        missing = sorted(set(record) - _columns(migrated_pg_conn, "prediction_logs"))
        assert missing == [], f"rekord save_node ma klucze bez kolumn prediction_logs: {missing}"


@pytest.mark.containers
class TestMigration061Digests:
    """MAIL-24: indeks prywatnego archiwum digestów — wiersz na cykl, upsert po
    `cycle_id`; bucket Storage tylko tam, gdzie jest schemat `storage` (Supabase)."""

    def test_digests_keep_one_row_per_cycle(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert {
            "cycle_id",
            "subject",
            "lead_json",
            "top_signals_json",
            "sent_at",
        } <= _columns(migrated_pg_conn, "digests")
        assert "idx_digests_sent_at" in _indexes(migrated_pg_conn, "digests")
        with _rolled_back(migrated_pg_conn) as cur:
            for subject in ("pierwszy", "ponowiony"):
                cur.execute(
                    "INSERT INTO digests (cycle_id, subject, lead_json, sent_at) "
                    "VALUES ('20261015T0530Z-1', %s, '[{\"symbol\": \"NVDA\"}]', NOW()) "
                    "ON CONFLICT (cycle_id) DO UPDATE SET subject = EXCLUDED.subject",
                    (subject,),
                )
            cur.execute("SELECT subject, top_signals_json FROM digests")
            assert cur.fetchall() == [("ponowiony", [])]

    def test_digests_are_closed_to_public_roles_and_open_to_service_role(
        self, supabase_like_conn
    ) -> None:
        with _rolled_back(supabase_like_conn) as cur:
            cur.execute("SELECT relrowsecurity FROM pg_class WHERE oid = 'digests'::regclass")
            assert cur.fetchone() == (True,)
            public = [_has_table_privilege(cur, role, "digests") for role in PUBLIC_API_ROLES]
            _set_role(cur, "service_role")
            cur.execute(
                "INSERT INTO digests (cycle_id, subject, sent_at) "
                "VALUES ('20261015T0530Z-SR', 's', NOW()) RETURNING id"
            )
            assert cur.fetchone() is not None

        assert public == [False, False]

    def test_061_is_idempotent_without_the_storage_schema(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        migration = (MIGRATIONS_DIR / "061_digests.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute("SELECT to_regclass('storage.buckets')")
            assert cur.fetchone() == (None,)


@pytest.mark.containers
class TestMigratedPgConn:
    """TEST-08: `migrated_pg_conn` ma komplet schematu niezależnie od kolejności
    testów — ten test przechodzi także uruchomiony sam (`-k`)."""

    def test_schema_is_complete_without_the_apply_test_running_first(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert {"id", "symbol", "price", "timestamp"} <= _columns(
            migrated_pg_conn, "price_snapshots"
        )
        assert "commitment_payload" in _columns(migrated_pg_conn, "prediction_logs")
        assert "idx_council_votes_prediction_investor_uniq" in _indexes(
            migrated_pg_conn, "council_votes"
        )


def test_migrations_live_where_the_supabase_cli_looks_for_them() -> None:
    """`supabase db push` czyta wyłącznie `supabase/migrations/`.

    Trzymanie plików gdzie indziej znaczy, że CLI zaaplikuje ZERO migracji
    i cicho powie „up to date" — najgorszy możliwy tryb awarii.
    """
    assert Path(__file__).parents[2] / "supabase" / "migrations" == MIGRATIONS_DIR
    assert MIGRATIONS_DIR.is_dir()


def test_migration_files_list_matches_the_directory() -> None:
    """Lista w teście i zawartość katalogu nie mogą się rozjechać.

    CLI globuje katalog, a ten moduł ma listę wpisaną ręcznie. Bez tej asercji
    nowa migracja mogłaby wejść na produkcję nieprzetestowana (albo odwrotnie:
    test aplikowałby plik, którego CLI nie widzi).
    """
    on_disk = sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))

    assert on_disk == MIGRATION_FILES


def test_migration_filenames_match_the_supabase_cli_pattern() -> None:
    """CLI parsuje nazwy regexem `^([0-9]+)_(.*)\\.sql$` (pkg/migration/file.go).

    Plik, który nie pasuje, jest POMIJANY z ostrzeżeniem na stderr — nie błędem.
    Dodatkowo: pierwsza migracja NIE MOŻE nazywać się `<14 cyfr>_init.sql`,
    bo CLI pomija taki plik dla wstecznej kompatybilności (list.go `shouldSkip`).
    """
    pattern = re.compile(r"^([0-9]+)_(.*)\.sql$")
    for name in MIGRATION_FILES:
        assert pattern.match(name), f"CLI pominąłby {name}"

    assert not re.match(r"^[0-9]{14}_init\.sql$", MIGRATION_FILES[0])


def test_migration_versions_sort_lexicographically_in_numeric_order() -> None:
    """CLI aplikuje pliki w kolejności `os.ReadDir` (leksykograficznej).

    Zero-padding do 3 cyfr sprawia, że kolejność leksykograficzna == numeryczna.
    Bez niego `10_x.sql` wypadłoby PRZED `2_x.sql` i migracje poszłyby na opak.
    """
    versions = [name.split("_", 1)[0] for name in MIGRATION_FILES]

    assert versions == sorted(versions)
    assert versions == [f"{int(v):03d}" for v in versions]


def _docker_without_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    import docker

    def _no_daemon() -> object:
        raise ConnectionError("brak demona Dockera")

    monkeypatch.setattr(docker, "from_env", _no_daemon)


def test_require_docker_turns_missing_daemon_into_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI-06: krok kontenerowy był zielony przy zerze wykonanych testów — brak
    demona dawał skip. Z `REQUIRE_DOCKER=1` (krok `containers` w ci.yml) to porażka.
    Bez markera `containers`: fabryka ma działać właśnie wtedy, gdy Dockera nie ma."""
    _docker_without_daemon(monkeypatch)
    monkeypatch.setenv("REQUIRE_DOCKER", "1")

    with pytest.raises(pytest.fail.Exception, match="REQUIRE_DOCKER"), _pgvector_connection():
        pass


def test_missing_docker_daemon_still_skips_without_require_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _docker_without_daemon(monkeypatch)
    monkeypatch.delenv("REQUIRE_DOCKER", raising=False)

    with pytest.raises(pytest.skip.Exception), _pgvector_connection():
        pass


@pytest.mark.containers
class TestMigration060ScorecardWeightsSha256:
    """CI-13: karta modelu niesie sha256 pliku wag z przebiegu; NULL bez zapisu wag."""

    def test_weights_hash_is_a_nullable_hex_digest(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert "weights_sha256" in _columns(migrated_pg_conn, "model_scorecards")
        digest = "0f" * 32
        with _rolled_back(migrated_pg_conn) as cur:
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, weights_sha256) VALUES "
                "('ALL', 'trained_successfully', %s), ('AAPL', 'evaluated_passed_gate', NULL)",
                (digest,),
            )
            cur.execute(
                "SELECT symbol, weights_sha256 FROM model_scorecards "
                "WHERE symbol IN ('ALL', 'AAPL') ORDER BY symbol"
            )
            rows = cur.fetchall()

        assert rows == [("AAPL", None), ("ALL", digest)]

    def test_weights_hash_rejects_anything_but_a_hex_digest(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        with (
            _rolled_back(migrated_pg_conn) as cur,
            pytest.raises(psycopg2.errors.CheckViolation),
        ):
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, weights_sha256) "
                "VALUES ('ALL', 'trained_successfully', 'sha256:abc')"
            )

    def test_060_is_idempotent(self, migrated_pg_conn: psycopg2.extensions.connection) -> None:
        migration = (MIGRATIONS_DIR / "060_model_scorecards_weights_sha256.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT count(*) FROM pg_constraint "
                "WHERE conrelid = 'model_scorecards'::regclass AND contype = 'c' "
                "AND pg_get_constraintdef(oid) LIKE '%weights_sha256%'"
            )
            assert cur.fetchone() == (1,)


@pytest.mark.containers
class TestMigration064ScorecardProvenance:
    """ML-22: karta modelu niesie proweniencję treningu; NULL dla pominięć i historii."""

    def test_provenance_columns_round_trip(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        assert {
            "dataset_sha256",
            "data_start",
            "data_end",
            "n_per_symbol",
            "training_params",
            "library_version",
            "feature_set_version",
            "git_sha",
            "label_overlap_count",
            "label_overlap_measured",
        } <= _columns(migrated_pg_conn, "model_scorecards")
        with _rolled_back(migrated_pg_conn) as cur:
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, dataset_sha256, data_start, "
                "data_end, n_per_symbol, training_params, library_version, "
                "feature_set_version, git_sha, label_overlap_count, label_overlap_measured) "
                "VALUES ('ALL', 'trained_successfully', %s, "
                "'2026-10-01T07:30:00+00:00', '2026-10-20T07:30:00+00:00', "
                "'{\"AAPL\": 40}'::jsonb, '{\"random_state\": 0}'::jsonb, "
                "'xgboost 3.2.0', 4, 'abc1234', 10, 49), ('AAPL', 'skipped', NULL, NULL, "
                "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                ("ab" * 32,),
            )
            cur.execute(
                "SELECT (n_per_symbol ->> 'AAPL')::int, "
                "(training_params ->> 'random_state')::int, feature_set_version, "
                "data_end > data_start, label_overlap_count, label_overlap_measured "
                "FROM model_scorecards WHERE symbol = 'ALL' AND dataset_sha256 IS NOT NULL"
            )
            row = cur.fetchone()

        assert row == (40, 0, 4, True, 10, 49)

    def test_dataset_hash_rejects_anything_but_a_hex_digest(
        self, migrated_pg_conn: psycopg2.extensions.connection
    ) -> None:
        with (
            _rolled_back(migrated_pg_conn) as cur,
            pytest.raises(psycopg2.errors.CheckViolation),
        ):
            cur.execute(
                "INSERT INTO model_scorecards (symbol, status, dataset_sha256) "
                "VALUES ('ALL', 'trained_successfully', 'not-a-digest')"
            )

    def test_064_is_idempotent(self, migrated_pg_conn: psycopg2.extensions.connection) -> None:
        migration = (MIGRATIONS_DIR / "064_model_scorecards_provenance.sql").read_text()
        with migrated_pg_conn.cursor() as cur:
            cur.execute(migration)
            cur.execute(migration)
            cur.execute(
                "SELECT count(*) FROM pg_constraint "
                "WHERE conrelid = 'model_scorecards'::regclass AND contype = 'c' "
                "AND pg_get_constraintdef(oid) LIKE '%dataset_sha256%'"
            )
            assert cur.fetchone() == (1,)
