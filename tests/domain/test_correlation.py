"""Testy domeny korelacji portfela — czysta logika, stdlib only.

Liczymy zwroty z cen, korelację Pearsona (z bramką na zerową wariancję)
oraz macierz korelacji + wykrywanie klastrów symboli poruszających się
razem. Zero zależności zewnętrznych — wszystko na `statistics`/`math`.
"""

from __future__ import annotations

import math

from src.domain.correlation import (
    correlation_matrix,
    detect_clusters,
    pearson,
    returns_from_prices,
)


class TestReturnsFromPrices:
    def test_simple_returns(self) -> None:
        # 100 → 110 → 121 to dwa zwroty +10%.
        assert returns_from_prices([100.0, 110.0, 121.0]) == [0.1, 0.1]

    def test_too_short_history_yields_empty(self) -> None:
        # Jeden punkt = brak zwrotów (nie da się policzyć zmiany).
        assert returns_from_prices([100.0]) == []
        assert returns_from_prices([]) == []

    def test_zero_price_does_not_blow_up(self) -> None:
        # Cena 0 w mianowniku — pomijamy ten krok zamiast dzielić przez zero.
        out = returns_from_prices([0.0, 100.0, 110.0])
        assert all(math.isfinite(r) for r in out)


class TestPearson:
    def test_perfectly_correlated_is_one(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]  # b = 2a → korelacja idealna
        assert pearson(a, b) == 1.0

    def test_anti_correlated_is_minus_one(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        assert pearson(a, b) == -1.0

    def test_zero_variance_returns_zero(self) -> None:
        # Jedna seria stała → wariancja 0 → nie ma korelacji (guard, nie NaN).
        a = [1.0, 1.0, 1.0, 1.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert pearson(a, b) == 0.0

    def test_mismatched_length_uses_common_prefix(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 99.0]
        b = [2.0, 4.0, 6.0, 8.0]
        assert pearson(a, b) == 1.0

    def test_too_short_returns_zero(self) -> None:
        assert pearson([1.0], [2.0]) == 0.0
        assert pearson([], []) == 0.0


class TestCorrelationMatrix:
    def test_builds_pairwise_entries(self) -> None:
        returns = {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.04, 0.06, 0.08],  # = 2*A
            "C": [-0.01, -0.02, -0.03, -0.04],  # = -A
        }
        matrix = correlation_matrix(returns)
        assert matrix[("A", "B")] == 1.0
        assert matrix[("A", "C")] == -1.0
        # Klucze posortowane leksykalnie — brak duplikatów (A,B)/(B,A).
        assert ("B", "A") not in matrix

    def test_skips_symbols_without_returns(self) -> None:
        returns = {
            "A": [0.01, 0.02, 0.03],
            "B": [0.02, 0.04, 0.06],
            "EMPTY": [],  # za krótka historia — pomijamy
        }
        matrix = correlation_matrix(returns)
        assert ("A", "EMPTY") not in matrix
        assert ("B", "EMPTY") not in matrix
        assert ("A", "B") in matrix


class TestDetectClusters:
    def test_groups_correlated_separates_uncorrelated(self) -> None:
        # A,B,C poruszają się razem (≥0.9); D jest niezależny.
        matrix = {
            ("A", "B"): 0.95,
            ("A", "C"): 0.92,
            ("B", "C"): 0.91,
            ("A", "D"): 0.10,
            ("B", "D"): 0.05,
            ("C", "D"): 0.00,
        }
        symbols = ["A", "B", "C", "D"]
        clusters = detect_clusters(matrix, symbols, threshold=0.9)
        # Tylko klaster ≥2 symboli jest istotny; D jako singleton pomijany.
        assert ("A", "B", "C") in clusters
        assert all(len(c) >= 2 for c in clusters)
        flat = {s for c in clusters for s in c}
        assert "D" not in flat

    def test_transitive_union_find(self) -> None:
        # A~B i B~C (ale A~C poniżej progu) — union-find i tak łączy A,B,C.
        matrix = {
            ("A", "B"): 0.95,
            ("B", "C"): 0.95,
            ("A", "C"): 0.50,
        }
        clusters = detect_clusters(matrix, ["A", "B", "C"], threshold=0.9)
        assert clusters == [("A", "B", "C")]

    def test_no_clusters_when_all_independent(self) -> None:
        matrix = {("A", "B"): 0.1, ("A", "C"): 0.2, ("B", "C"): 0.0}
        assert detect_clusters(matrix, ["A", "B", "C"], threshold=0.9) == []

    def test_members_sorted_within_cluster(self) -> None:
        matrix = {("X", "A"): 0.99}
        clusters = detect_clusters(matrix, ["X", "A"], threshold=0.9)
        assert clusters == [("A", "X")]
