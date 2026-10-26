"""DOC-27: reguła TDD z CLAUDE.md jako sprawdzalna polityka gałęzi.

Commit `fix:` bez zmiany w tests/ i bez wcześniejszego commita `test:` na tej samej gałęzi
jest flagowany. Poprawki samego CI i dokumentacji (`fix(ci):`, `fix(docs):`) nie mają
testu do napisania, a merge commity nie są zmianą — oba pomijamy, inaczej polityka byłaby
fałszywie czerwona.
"""

from __future__ import annotations

from src.tools.commit_policy import CommitInfo, fix_commits_without_test, parse_git_log


def test_fix_commit_without_test_change_is_flagged() -> None:
    assert fix_commits_without_test([CommitInfo("fix: x", ["src/a.py"])]) == ["fix: x"]
    assert (
        fix_commits_without_test(
            [
                CommitInfo("test: reproduce x", ["tests/t.py"]),
                CommitInfo("fix: x", ["src/a.py"]),
            ]
        )
        == []
    )


def test_fix_commit_that_changes_a_test_passes() -> None:
    commits = [CommitInfo("fix(report): x (MAIL-1)", ["src/a.py", "tests/application/t.py"])]

    assert fix_commits_without_test(commits) == []


def test_a_test_commit_after_the_fix_does_not_cover_it() -> None:
    commits = [
        CommitInfo("fix: x", ["src/a.py"]),
        CommitInfo("test: reproduce x", ["tests/t.py"]),
    ]

    assert fix_commits_without_test(commits) == ["fix: x"]


def test_ci_and_docs_fixes_are_exempt_but_other_scopes_are_not() -> None:
    commits = [
        CommitInfo("fix(ci): pin the action", [".github/workflows/ci.yml"]),
        CommitInfo("fix(docs): broken link", ["docs/a.md"]),
        CommitInfo("fix(ci,docs): both", ["docs/a.md"]),
        CommitInfo("fix(ci,graph): mixed", ["src/a.py"]),
        CommitInfo("fix(graph)!: breaking", ["src/a.py"]),
    ]

    assert fix_commits_without_test(commits) == ["fix(ci,graph): mixed", "fix(graph)!: breaking"]


def test_merge_commits_and_other_types_are_ignored() -> None:
    commits = [
        CommitInfo("Merge branch 'fix/x' into main", ["src/a.py"]),
        CommitInfo("feat: y", ["src/b.py"]),
        CommitInfo("refactor: z", ["src/c.py"]),
        CommitInfo("fixup: not a conventional fix", ["src/d.py"]),
    ]

    assert fix_commits_without_test(commits) == []


def test_parse_git_log_reads_subjects_and_files_in_order() -> None:
    output = (
        "\x1etest: reproduce x\n\ntests/t.py\n"
        "\x1efix: x\n\nsrc/a.py\nsrc/b.py\n"
        "\x1edocs: no files\n"
    )

    assert parse_git_log(output) == [
        CommitInfo("test: reproduce x", ("tests/t.py",)),
        CommitInfo("fix: x", ("src/a.py", "src/b.py")),
        CommitInfo("docs: no files", ()),
    ]
