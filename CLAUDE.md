# CLAUDE.md — dodatki Claude Code w repo StockAgent

Wszystkie zasady pracy w repo są w `AGENTS.md`, jedynym źródle dla każdego agenta (DOC-29).
Claude Code wkleja go w miejsce linii poniżej. Nową regułę dopisuj w `AGENTS.md`; ten plik
trzyma wyłącznie to, co dotyczy samego Claude Code.

@AGENTS.md

## Dodatki Claude Code

- Gdy import się nie rozwinął (w kontekście nie ma sekcji `AGENTS.md`, np. w harnessie bez
  importów `@`), przeczytaj `AGENTS.md` przed pierwszą zmianą.
- Nie uruchamiaj `/init` na tym repo: nadpisałby ten plik pełną kopią zasad, a druga kopia to
  dokładnie ten dryf, który usunął DOC-29.
- Pamięć Claude'a (auto-memory poza repo) nie jest źródłem zasad repo: regułę, która ma
  obowiązywać każdego agenta, dopisz do `AGENTS.md`.
- Brief subagenta nie musi powtarzać zasad: subagent w tym repo ładuje `CLAUDE.md`, a z nim
  `AGENTS.md`. Podaj mu zadanie, kryteria akceptacji i pliki.
