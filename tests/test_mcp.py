"""Integration tests for the MCP tool layer.

These call the tool functions directly rather than over stdio — the transport is
the SDK's problem, the tool behaviour is ours. The stdio handshake is covered by
`test_server_exposes_expected_tools`, which inspects the registered tool set.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="needs the mcp extra")

from atlas import mcp_server as m


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    """Point both scopes at tmp_path so tests never touch the real stores."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".atlas").mkdir()
    (project / "handbook.md").write_text(
        "# Handbook\n\nThe deploy job requires an approved release tag.\n"
    )
    monkeypatch.setenv("ATLAS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ATLAS_PROJECT", str(project))
    return project


def test_server_exposes_expected_tools():
    names = {t.name for t in m.server._tool_manager.list_tools()}
    assert {
        "atlas_recall",
        "atlas_remember",
        "atlas_facts",
        "atlas_forget",
        "atlas_pin",
        "atlas_index",
        "atlas_status",
    } <= names


def test_tools_have_descriptions():
    for tool in m.server._tool_manager.list_tools():
        assert tool.description and len(tool.description) > 40, (
            f"{tool.name} needs a description that tells an agent when to call it"
        )


def test_remember_then_recall_roundtrip():
    m.atlas_remember("Integration tests need DOCKER_HOST set", tags=["testing"])
    out = m.atlas_recall("docker host for tests")
    assert "DOCKER_HOST" in out


def test_remember_reports_dedupe():
    m.atlas_remember("The API gateway rate limits at 100 rps")
    again = m.atlas_remember("the API gateway rate limits at 100 rps")
    assert "Updated existing fact" in again


def test_recall_miss_is_actionable():
    out = m.atlas_recall("something nobody has ever written down")
    assert "No matches" in out
    assert "atlas_remember" in out, "a miss should tell the agent what to do next"


def test_index_then_recall_file_content(isolated_stores):
    result = m.atlas_index(str(isolated_stores))
    assert "Indexed" in result

    out = m.atlas_recall("approved release tag")
    assert "handbook.md" in out


def test_pin_surfaces_fact_on_unrelated_query():
    out = m.atlas_remember("Never commit directly to main")
    fact_id = int("".join(c for c in out.split("fact")[1] if c.isdigit()))
    m.atlas_pin(fact_id, True)

    assert "Never commit directly to main" in m.atlas_recall("unrelated query about parsers")


def test_forget_removes_fact():
    out = m.atlas_remember("A temporary note")
    fact_id = int("".join(c for c in out.split("fact")[1] if c.isdigit()))

    assert "Forgot" in m.atlas_forget(fact_id)
    assert "No fact" in m.atlas_forget(fact_id)


def test_hostile_query_does_not_crash():
    for hostile in ['unbalanced "quote', "AND OR", "*", "((("]:
        assert isinstance(m.atlas_recall(hostile), str)


def test_index_missing_path_is_reported():
    assert "does not exist" in m.atlas_index("/nope/definitely/not/here")


def test_status_reports_empty_state():
    assert "Nothing indexed yet" in m.atlas_status()
