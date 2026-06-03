"""Globalna konfiguracja testów.

Izolacja od `config.toml`: testy jednostkowe asertują czyste defaulty pól
`Settings` oraz przekazują config przez jawne kwargs / fixture'y. Gdyby źródło
TOML było aktywne, `config.toml` z repo nadpisywałby defaulty (np.
`nbp_enabled=true`, `crypto_symbols=[BTC,ETH]`) i psuł te asercje.

Flagę ustawiamy na poziomie importu — ZANIM jakikolwiek test zbuduje `Settings`.
Testy, które celowo weryfikują ładowanie z TOML (`test_config_toml.py`),
zdejmują ją lokalnie przez `monkeypatch.delenv`.
"""

import os

os.environ["STOCKAGENT_DISABLE_TOML"] = "1"
