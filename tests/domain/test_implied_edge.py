"""Testy domeny "edge vs rynek opcji" — dywergencja predykcji od implied move.

Czysta logika, stdlib only (float + math.sqrt, jak w correlation.py). Sprawdzamy
sqrt-time scaling annualizowanej IV na horyzont godzinowy, klasyfikację jakościową
(MODEL_AHEAD / MARKET_AHEAD / ALIGNED / NO_SIGNAL) oraz twarde bramki na brak IV,
IV <= 0 i horyzont <= 0 (nigdy dzielenie przez zero).

UWAGA metodyczna (zamrożona w tym teście): `edge_sigma` to sygnał INFORMACYJNY do
walidacji na zamkniętych predykcjach — NIE MA jeszcze roli decyzyjnej. Testy celowo
nie wiążą etykiety z żadną akcją tradingową.
"""

from __future__ import annotations

import math

import pytest

from src.domain.implied_edge import (
    EdgeLabel,
    ImpliedEdge,
    evaluate_edge,
    implied_move_12h,
    model_move_from_prices,
)

# Godziny kalendarzowe w roku — baza annualizacji sqrt-time.
_HOURS_PER_YEAR = 365.0 * 24.0


class TestImpliedMove12h:
    def test_sqrt_time_scaling_from_annualized_iv(self) -> None:
        # 45% annualizowanej IV przeskalowane na 12h kalendarzowych.
        expected = 0.45 * math.sqrt(12.0 / _HOURS_PER_YEAR)
        assert implied_move_12h(0.45, 12.0) == pytest.approx(expected)

    def test_dwa_razy_dluzszy_horyzont_to_sqrt2_razy_wiekszy_ruch(self) -> None:
        # sqrt-time: podwojenie horyzontu mnoży ruch przez sqrt(2), nie przez 2.
        move_12h = implied_move_12h(0.30, 12.0)
        move_24h = implied_move_12h(0.30, 24.0)
        assert move_24h == pytest.approx(move_12h * math.sqrt(2.0))

    def test_zero_iv_daje_zerowy_ruch(self) -> None:
        assert implied_move_12h(0.0, 12.0) == 0.0

    def test_zero_horyzont_daje_zerowy_ruch(self) -> None:
        assert implied_move_12h(0.45, 0.0) == 0.0


class TestModelMoveFromPrices:
    def test_procentowy_ruch_model_od_ceny_biezacej(self) -> None:
        # 100 → 103.1 to +3.1% przewidywanego ruchu.
        assert model_move_from_prices(100.0, 103.1) == pytest.approx(0.031)

    def test_spadkowa_predykcja_ma_znak_ujemny(self) -> None:
        assert model_move_from_prices(200.0, 190.0) == pytest.approx(-0.05)

    def test_zerowa_cena_biezaca_daje_zero(self) -> None:
        # Brak punktu odniesienia — zero zamiast dzielenia przez zero.
        assert model_move_from_prices(0.0, 103.1) == 0.0


class TestEvaluateEdge:
    def test_model_ahead_gdy_model_widzi_wiecej_niz_rynek(self) -> None:
        # "model: +3.1%, opcje wyceniają ±1.2% → edge ~2.6σ" — model wie coś,
        # czego rynek nie wycenił.
        result = evaluate_edge(model_move=0.031, iv=0.324, horizon_hours=12.0)
        assert isinstance(result, ImpliedEdge)
        assert result.label is EdgeLabel.MODEL_AHEAD
        assert result.edge_sigma is not None
        assert result.edge_sigma == pytest.approx(
            abs(0.031) / implied_move_12h(0.324, 12.0)
        )
        assert result.edge_sigma > 1.5

    def test_market_ahead_gdy_rynek_wycenia_ruch_a_model_spi(self) -> None:
        # "opcje wyceniają ±6%, model śpi — rynek wie coś, czego nie widzimy".
        result = evaluate_edge(model_move=0.002, iv=0.90, horizon_hours=12.0)
        assert result.label is EdgeLabel.MARKET_AHEAD
        assert result.edge_sigma is not None
        assert result.edge_sigma < 0.5

    def test_aligned_gdy_model_i_rynek_zgodni(self) -> None:
        # Ruch modelu ~= implied move → brak edge'u, zgodność.
        implied = implied_move_12h(0.45, 12.0)
        result = evaluate_edge(model_move=implied, iv=0.45, horizon_hours=12.0)
        assert result.label is EdgeLabel.ALIGNED
        assert result.edge_sigma == pytest.approx(1.0)

    def test_brak_iv_daje_no_signal(self) -> None:
        result = evaluate_edge(model_move=0.031, iv=None, horizon_hours=12.0)
        assert result.label is EdgeLabel.NO_SIGNAL
        assert result.edge_sigma is None
        assert result.implied_move is None

    def test_iv_niedodatnia_daje_no_signal(self) -> None:
        # IV <= 0 nie ma sensu — brak sygnału, żadnego dzielenia przez zero.
        assert evaluate_edge(0.031, 0.0, 12.0).label is EdgeLabel.NO_SIGNAL
        assert evaluate_edge(0.031, -0.4, 12.0).label is EdgeLabel.NO_SIGNAL

    def test_horyzont_niedodatni_daje_no_signal(self) -> None:
        assert evaluate_edge(0.031, 0.45, 0.0).label is EdgeLabel.NO_SIGNAL
        assert evaluate_edge(0.031, 0.45, -12.0).label is EdgeLabel.NO_SIGNAL

    def test_znak_predykcji_nie_zmienia_wielkosci_edge(self) -> None:
        # edge_sigma to WIELKOŚĆ dywergencji — |model_move| / implied_move.
        up = evaluate_edge(model_move=0.031, iv=0.324, horizon_hours=12.0)
        down = evaluate_edge(model_move=-0.031, iv=0.324, horizon_hours=12.0)
        assert up.edge_sigma == down.edge_sigma
        assert up.label is down.label

    def test_edge_jest_sygnalem_informacyjnym_nie_decyzja(self) -> None:
        # DOKUMENTACJA KONTRAKTU: ImpliedEdge niesie etykietę + surową wartość,
        # ale NIE eksponuje żadnej rekomendacji akcji (buy/sell/hold/gate).
        # To sygnał do walidacji na zamkniętych predykcjach, nie sterownik decyzji.
        result = evaluate_edge(model_move=0.031, iv=0.324, horizon_hours=12.0)
        public = {f for f in vars(result) if not f.startswith("_")}
        assert public == {"label", "edge_sigma", "implied_move", "model_move"}
        for banned in ("action", "signal", "recommendation", "decision", "gate"):
            assert not hasattr(result, banned)
