# The report email

What lands in the inbox and why each section is there. The pipeline that produces it is in [how-it-works.md](how-it-works.md).

## Email anatomy

The email is a **17 kB+ structured digest in Polish** (HTML + plain-text fallback):

- 🕒 **NYSE session** — open / premarket / after-hours / weekend (`zoneinfo` America/New_York)
- 📊 **Three summary boxes** — predictions / ignored / errors
- 🎯 **Strongest signals** — top BUY/SELL, colour-coded (priority 1: actionable)
- 🚨 **Warning signals** — divergence + AV/LLM conflict + low signal
- 📊 **Portfolio mood** — average sentiment, most positive/negative, high-confidence count
- 📊 **Closed predictions (24h)** — full **post-mortem** per resolved prediction (stocks **and** crypto): the original thesis (*why up/down* — `reasoning_text`), the actual move (`$A → $B (+C%)`), the ✅ Trafiona / ❌ Błędna verdict, and **why** (the `reflect` diagnosis for misses, thesis-confirmed for hits). All from data already in `prediction_logs` — no extra LLM calls.
- 🎯 **Accuracy history** — directional hit-rate over the last 30 days
- 🪙 **Krypto** — dedicated section listing every tracked crypto ticker (price · Δ · trend/forecast for `saved`, "poniżej progu" for `ignored`). Rendered from the domain asset class (`Asset.asset_type`, surfaced as `SymbolResult.asset_class`), **independent of the 5% volatility gate** — so crypto stays visible even on a quiet day instead of vanishing into the "Pominięte" chips.
- 📈 **Δ chart** — inline div bars, change since last cycle
- 🧠 **Self-Reflection** — lessons learned per symbol (purple box)
- 🔮 **Predictions table** — price · Δ (per cycle) · trend · forecast (next cycle) `+X.YZ%` · confidence · sentiment · news count. Each row is tagged with its **sector** next to the company name (stocks → Polish sector e.g. `Cyberbezpieczeństwo`, ETFs → `ETF`, crypto → `Krypto`); mapping in `report_formatting.SECTORS`.
- 💡 **Worth a look (sectors in motion)** — peer suggestions computed **from the current cycle only, zero extra API calls**: when a stock sector runs hot (its strongest monitored move `|Δ| ≥ 3%` or mean `|sentiment| ≥ 0.3`), the report suggests notable peers in that sector you don't yet monitor (curated `report_formatting.PEERS`), e.g. *Cyberbezpieczeństwo gorące (CRWD +6.0%, PANW +4.2%) — rozważ: ZS, FTNT, CYBR*. Top 3 sectors, 3 peers each; section hidden when nothing is hot.
- 📈 **Forecast chart** — inline div bars
- 💡 **Reasoning** + 📰 **Top news** (clickable `<a href>` links to the original articles)
- 📈 **Sentiment vs price correlation** (scatter plot)
- ⏸ **Ignored** (chip list) + ⚠️ **Errors**

All sections are **conditional** — they only render when data exists. The first cold-start cycle produces a concise report; once you have 50+ cycles behind you, accuracy sparklines and history kick in.
