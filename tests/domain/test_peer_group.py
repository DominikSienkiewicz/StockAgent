from src.domain.peer_group import build_peer_context, peers_of


class TestPeersOf:
    """`peers_of` zwraca WSZYSTKIE inne symbole dzielące jakąkolwiek grupę
    z danym symbolem (unia po grupach), bez samego symbolu, z dedupem i
    stabilną kolejnością."""

    def test_returns_peers_sharing_group(self):
        groups = {"semis": ["NVDA", "AMD", "TSM"]}
        # NVDA dzieli grupę "semis" z AMD i TSM.
        assert peers_of("NVDA", groups) == ("AMD", "TSM")

    def test_excludes_self(self):
        groups = {"semis": ["NVDA", "AMD", "TSM"]}
        peers = peers_of("AMD", groups)
        assert "AMD" not in peers
        assert peers == ("NVDA", "TSM")

    def test_unions_peers_across_multiple_groups(self):
        # AAPL należy do "mega_tech" ORAZ "fruit" — unia peerów z obu grup.
        groups = {
            "mega_tech": ["AAPL", "MSFT", "GOOG"],
            "fruit": ["AAPL", "BANANA"],
        }
        peers = peers_of("AAPL", groups)
        # Stabilna kolejność: najpierw peery z "mega_tech", potem z "fruit".
        assert peers == ("MSFT", "GOOG", "BANANA")

    def test_dedups_peer_appearing_in_two_shared_groups(self):
        # MSFT współdzieli z AAPL dwie grupy — w wyniku ma wystąpić raz.
        groups = {
            "mega_tech": ["AAPL", "MSFT"],
            "cloud": ["AAPL", "MSFT"],
        }
        assert peers_of("AAPL", groups) == ("MSFT",)

    def test_symbol_in_no_group_returns_empty(self):
        groups = {"semis": ["NVDA", "AMD"]}
        assert peers_of("AAPL", groups) == ()

    def test_empty_groups_returns_empty(self):
        assert peers_of("NVDA", {}) == ()

    def test_lone_member_group_has_no_peers(self):
        groups = {"solo": ["NVDA"]}
        assert peers_of("NVDA", groups) == ()

    def test_returns_tuple(self):
        peers = peers_of("NVDA", {"semis": ["NVDA", "AMD"]})
        assert isinstance(peers, tuple)


class TestBuildPeerContext:
    """`build_peer_context` zwraca pary (peer_symbol, stance) tylko dla tych
    peerów danego symbolu, które JUŻ przetworzono w tym cyklu (są w `processed`),
    stabilnie uporządkowane. Pusto, gdy żaden peer nie był jeszcze liczony."""

    def test_includes_only_processed_peers(self):
        groups = {"semis": ["NVDA", "AMD", "TSM"]}
        # W tym cyklu policzono już AMD, ale jeszcze nie TSM.
        processed = {"AMD": "BUY"}
        context = build_peer_context("NVDA", groups, processed)
        assert context == (("AMD", "BUY"),)

    def test_excludes_unprocessed_peers(self):
        groups = {"semis": ["NVDA", "AMD", "TSM"]}
        # Tylko TSM policzony — AMD jeszcze nie, więc nie trafia do kontekstu.
        processed = {"TSM": "HOLD"}
        context = build_peer_context("NVDA", groups, processed)
        symbols = [sym for sym, _ in context]
        assert "AMD" not in symbols
        assert context == (("TSM", "HOLD"),)

    def test_excludes_self_even_if_in_processed(self):
        groups = {"semis": ["NVDA", "AMD"]}
        # NVDA mógł być wstępnie zaznaczony w processed — nie jest własnym peerem.
        processed = {"NVDA": "SELL", "AMD": "BUY"}
        context = build_peer_context("NVDA", groups, processed)
        symbols = [sym for sym, _ in context]
        assert "NVDA" not in symbols
        assert context == (("AMD", "BUY"),)

    def test_stable_order_follows_peer_order(self):
        # Kolejność wyniku idzie za kolejnością peerów (jak w peers_of),
        # niezależnie od kolejności wstawienia do `processed`.
        groups = {"mega_tech": ["AAPL", "MSFT", "GOOG"]}
        processed = {"GOOG": "BUY", "MSFT": "HOLD"}
        context = build_peer_context("AAPL", groups, processed)
        assert context == (("MSFT", "HOLD"), ("GOOG", "BUY"))

    def test_empty_when_no_processed_peers(self):
        groups = {"semis": ["NVDA", "AMD", "TSM"]}
        # Pierwszy symbol w cyklu — nikt z peerów jeszcze nie policzony.
        assert build_peer_context("NVDA", groups, {}) == ()

    def test_empty_when_symbol_has_no_peers(self):
        groups = {"semis": ["NVDA", "AMD"]}
        processed = {"NVDA": "BUY", "MSFT": "SELL"}
        # AAPL nie ma peerów w tej konfiguracji.
        assert build_peer_context("AAPL", groups, processed) == ()

    def test_unions_processed_peers_across_groups(self):
        groups = {
            "mega_tech": ["AAPL", "MSFT"],
            "fruit": ["AAPL", "BANANA"],
        }
        processed = {"MSFT": "HOLD", "BANANA": "BUY"}
        context = build_peer_context("AAPL", groups, processed)
        assert context == (("MSFT", "HOLD"), ("BANANA", "BUY"))

    def test_returns_tuple(self):
        context = build_peer_context(
            "NVDA", {"semis": ["NVDA", "AMD"]}, {"AMD": "BUY"}
        )
        assert isinstance(context, tuple)
