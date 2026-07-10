"""Testy migracji SQL z prawdziwym Postgres + pgvector (Docker).

Uruchamia kontener `pgvector/pgvector:pg16`, aplikuje wszystkie migracje
w kolejności i weryfikuje schemat — w szczególności kolumnę `embedding`,
której brak powodował błędy PGRST204 na produkcji.

Uruchomienie:
    uv run pytest -m containers
"""

from __future__ import annotations

from pathlib import Path

import psycopg2
import pytest
from testcontainers.postgres import PostgresContainer

MIGRATIONS_DIR = Path(__file__).parents[2] / "migrations"
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
