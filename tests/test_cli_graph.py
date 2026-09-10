"""The command line, exercised the way a user does.

`atlas <folder>` is the headline flow, so the argument rewriting that makes the
bare form work gets its own tests — including the case where it must *not* fire.
"""

from __future__ import annotations

import json

import pytest

from atlas.cli import _expand_default_command, build_parser, main


@pytest.fixture(autouse=True)
def _known_commands():
    """`_expand_default_command` reads a set that build_parser populates."""
    build_parser()


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr()


# -- the bare form ---------------------------------------------------------


def test_a_directory_argument_becomes_build(tmp_path):
    assert _expand_default_command([str(tmp_path)]) == ["build", str(tmp_path)]
    assert _expand_default_command([str(tmp_path), "--ocr"]) == ["build", str(tmp_path), "--ocr"]


def test_explicit_commands_and_flags_are_left_alone(tmp_path):
    assert _expand_default_command(["build", str(tmp_path)]) == ["build", str(tmp_path)]
    assert _expand_default_command(["--version"]) == ["--version"]
    assert _expand_default_command([]) == []


def test_a_mistyped_command_is_not_read_as_a_path():
    """`atlas serach foo` must produce argparse's error, not a confusing one."""
    assert _expand_default_command(["serach", "foo"]) == ["serach", "foo"]


def test_a_nonexistent_directory_is_not_rewritten():
    assert _expand_default_command(["./does-not-exist"]) == ["./does-not-exist"]


# -- build -----------------------------------------------------------------


def test_build_reports_what_it_did(corpus, capsys):
    code, out = run([str(corpus)], capsys)
    assert code == 0
    assert "Building graph from" in out.out
    assert "files" in out.out and "nodes" in out.out
    assert (corpus / ".atlas" / "store.db").exists()


def test_build_rejects_a_file(tmp_path, capsys):
    target = tmp_path / "a.md"
    target.write_text("x")
    code, out = run(["build", str(target)], capsys)
    assert code == 1
    assert "folder that contains it" in out.err


def test_build_rejects_a_missing_folder(tmp_path, capsys):
    code, out = run(["build", str(tmp_path / "nope")], capsys)
    assert code == 1
    assert "does not exist" in out.err


# -- querying --------------------------------------------------------------


@pytest.fixture
def graph_folder(corpus, capsys):
    main([str(corpus), "-q"])
    capsys.readouterr()
    return corpus


def test_search_finds_content(graph_folder, capsys):
    code, out = run(["search", "migration", "--store", str(graph_folder)], capsys)
    assert code == 0
    assert "brief.md" in out.out


def test_search_json_is_machine_readable(graph_folder, capsys):
    code, out = run(["search", "invoice", "--store", str(graph_folder), "--json"], capsys)
    payload = json.loads(out.out)
    assert code == 0 and payload
    assert {"uid", "kind", "name", "props", "preview", "score"} <= set(payload[0])


def test_node_shows_properties(graph_folder, capsys):
    code, out = run(["node", "doc:docs/report.pdf", "--store", str(graph_folder)], capsys)
    assert code == 0
    assert "pages" in out.out and "modality" in out.out


def test_missing_node_fails_cleanly(graph_folder, capsys):
    code, out = run(["node", "doc:nope", "--store", str(graph_folder)], capsys)
    assert code == 1
    assert "no node with uid" in out.err


def test_neighbors_shows_direction(graph_folder, capsys):
    code, out = run(
        ["neighbors", "ent:identifier:inv-2024-0912", "--store", str(graph_folder),
         "--direction", "in"],
        capsys,
    )
    assert code == 0
    assert out.out.count("←") >= 3, "three files mention this invoice"


def test_path_prints_a_chain(graph_folder, capsys):
    code, out = run(
        ["path", "doc:README.md", "doc:docs/report.pdf", "--store", str(graph_folder)], capsys
    )
    assert code == 0
    assert "README" in out.out and "report" in out.out


def test_hubs_and_graph_stats(graph_folder, capsys):
    code, out = run(["hubs", "--store", str(graph_folder), "-k", "3"], capsys)
    assert code == 0 and "most connected" in out.out

    code, out = run(["graph", "--store", str(graph_folder), "--json"], capsys)
    payload = json.loads(out.out)
    assert code == 0
    assert payload["counts"]["nodes"] > 0


def test_export_is_a_loadable_node_link_graph(graph_folder, tmp_path, capsys):
    out_file = tmp_path / "graph.json"
    code, _ = run(["export", "--store", str(graph_folder), "--out", str(out_file)], capsys)
    assert code == 0

    payload = json.loads(out_file.read_text())
    uids = {n["uid"] for n in payload["nodes"]}
    assert uids and payload["edges"]
    # Every edge endpoint must exist, or the file will not load anywhere.
    for edge in payload["edges"]:
        assert edge["source"] in uids and edge["target"] in uids


# -- diagnostics -----------------------------------------------------------


def test_querying_without_a_graph_explains_how_to_make_one(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT", str(tmp_path))
    code, out = run(["hubs"], capsys)
    assert code == 1
    assert "atlas" in out.err and "Build one" in out.err


def test_doctor_lists_capabilities(capsys):
    code, out = run(["doctor"], capsys)
    assert code == 0
    assert "pypdf" in out.out and "ffmpeg" in out.out
