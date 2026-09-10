from __future__ import annotations

import pytest

from atlas.store import Store


@pytest.fixture
def project_store(tmp_path):
    store = Store(tmp_path / ".atlas" / "store.db", "project")
    yield store
    store.close()


@pytest.fixture
def global_store(tmp_path):
    store = Store(tmp_path / "global" / "global.db", "global")
    yield store
    store.close()


@pytest.fixture
def repo(tmp_path):
    """A small tree with enough shape to exercise chunking and recall."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()

    (root / "README.md").write_text(
        "# Widget Service\n\n"
        "## Running tests\n\n"
        "Tests need a live Redis. Start it with `make redis` before pytest.\n\n"
        "## Deploying\n\n"
        "Deploys go out through the release pipeline, never manually.\n"
    )
    (root / "src" / "auth.py").write_text(
        "class TokenValidator:\n"
        "    \"\"\"Validates bearer tokens against the session store.\"\"\"\n\n"
        "    def validate(self, token):\n"
        "        return self.sessions.lookup(token) is not None\n"
    )
    (root / "src" / "billing.py").write_text(
        "def compute_invoice_total(items):\n"
        "    return sum(item.price for item in items)\n"
    )
    (root / "docs" / "architecture.md").write_text(
        "# Architecture\n\nThe gateway fans out to three services over gRPC.\n"
    )
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    return root


@pytest.fixture
def graph(project_store):
    """An empty graph sharing the project store's connection."""
    from atlas.graph import GraphStore

    return GraphStore(project_store.conn)


@pytest.fixture
def corpus(tmp_path):
    """A small multimodal folder: markdown, PDF, CSV, image, audio."""
    import fixtures

    return fixtures.make_corpus(tmp_path / "corpus")


@pytest.fixture
def built(corpus, tmp_path):
    """`corpus`, converted to a graph. Yields (root, store, graph)."""
    from atlas.build import build
    from atlas.extract import ExtractOptions
    from atlas.graph import GraphStore
    from atlas.store import Store

    store = Store(corpus / ".atlas" / "store.db", "project")
    graph_store = GraphStore(store.conn)
    build(corpus, store, graph_store, ExtractOptions(transcribe=False, keyframes=0))
    yield corpus, store, graph_store
    store.close()
