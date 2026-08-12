from __future__ import annotations

import json

import pytest

from atlas.cli import main


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "docs").mkdir(parents=True)
    (project / "docs" / "ops.md").write_text(
        "# Operations\n\nRotate credentials with `make rotate-creds` every quarter.\n"
    )
    monkeypatch.setenv("ATLAS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ATLAS_PROJECT", str(project))
    monkeypatch.chdir(project)
    return project


def run(*argv) -> int:
    return main(list(argv))


def test_init_creates_store_and_indexes(isolated, capsys):
    assert run("init") == 0
    assert (isolated / ".atlas" / "store.db").exists()
    assert "indexed" in capsys.readouterr().out


def test_remember_and_facts(capsys):
    run("init", "--no-index")
    assert run("remember", "Rotate creds quarterly", "--tag", "ops") == 0
    run("facts")
    out = capsys.readouterr().out
    assert "Rotate creds quarterly" in out
    assert "#ops" in out


def test_recall_json_is_machine_readable(capsys):
    run("init")
    run("remember", "Credentials rotate every quarter")
    capsys.readouterr()

    assert run("recall", "rotate credentials", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert {"kind", "scope", "score", "ref", "text"} <= set(payload[0])


def test_recall_reports_empty_state_gracefully(capsys):
    run("init", "--no-index")
    capsys.readouterr()
    assert run("recall", "anything at all") == 0
    assert "No matches" in capsys.readouterr().out


def test_facts_export_renders_markdown(capsys):
    run("init", "--no-index")
    run("remember", "Never force push to main", "--pin")
    capsys.readouterr()

    run("facts", "--export")
    out = capsys.readouterr().out
    assert out.startswith("# Project memory")
    assert "## Always" in out


def test_pin_and_unpin(capsys):
    run("init", "--no-index")
    run("remember", "Use conventional commits")
    capsys.readouterr()

    assert run("pin", "1") == 0
    assert "Pinned" in capsys.readouterr().out
    assert run("unpin", "1") == 0
    assert "Unpinned" in capsys.readouterr().out


def test_forget_missing_fact_exits_nonzero(capsys):
    run("init", "--no-index")
    assert run("forget", "999") == 1


def test_status_lists_sources(isolated, capsys):
    run("init")
    capsys.readouterr()
    run("status")
    out = capsys.readouterr().out
    assert str(isolated) in out
    assert "chunks" in out


def test_sources_remove(isolated, capsys):
    run("init")
    capsys.readouterr()
    assert run("sources", "--remove", str(isolated)) == 0
    capsys.readouterr()

    run("sources")
    assert str(isolated) not in capsys.readouterr().out


def test_global_scope_write_is_separate(capsys):
    run("init", "--no-index")
    run("remember", "Prefer uv over pip", "--scope", "global")
    capsys.readouterr()

    run("facts", "--scope", "project")
    assert "Prefer uv" not in capsys.readouterr().out

    run("facts", "--scope", "global")
    assert "Prefer uv" in capsys.readouterr().out


def test_reindex_without_sources_is_not_an_error(capsys):
    assert run("reindex") == 0
    assert "Nothing indexed yet" in capsys.readouterr().out


def test_embed_without_extra_reports_cleanly(capsys):
    pytest.importorskip("atlas.embeddings")
    from atlas import embeddings

    if embeddings.load_embedder() is not None:
        pytest.skip("embeddings extra is installed")

    run("init", "--no-index")
    assert run("embed") == 1
