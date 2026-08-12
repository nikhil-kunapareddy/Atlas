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
