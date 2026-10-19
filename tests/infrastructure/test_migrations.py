"""Testy migracji SQL z prawdziwym Postgres + pgvector (Docker).

Uruchamia kontener `pgvector/pgvector:pg16`, aplikuje wszystkie migracje
w kolejności i weryfikuje schemat — w szczególności kolumnę `embedding`,
której brak powodował błędy PGRST204 na produkcji.

Uruchomienie:
    uv run pytest -m containers
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import psycopg2
import pytest
from testcontainers.postgres import PostgresContainer

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
]

PGVECTOR_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_conn():
    """Kontener Postgres z pgvector — żyje przez cały moduł testów.

    Skipuje cały moduł gdy Docker nie jest dostępny (CI bez demona, lokalnie
    bez Docker Desktop).
    """
    import docker

    try:
        docker.from_env().ping()
    except Exception:
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
        yield conn
        conn.close()


def _apply_migrations(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        for filename in MIGRATION_FILES:
            sql = (MIGRATIONS_DIR / filename).read_text()
            cur.execute(sql)


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


@pytest.fixture
def fresh_pg():
    """Świeży, IZOLOWANY kontener Postgres (function scope) — dla testów, które
    muszą zacząć od stanu PRZED konkretną migracją (nie da się tego uzyskać w
    module-scoped kontenerze, gdzie wszystkie migracje są już zaaplikowane)."""
    import docker

    try:
        docker.from_env().ping()
    except Exception:
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
        yield conn
        conn.close()


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
