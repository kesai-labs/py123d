"""Test that the Sphinx documentation builds without errors."""

import subprocess
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


@pytest.fixture(autouse=True)
def _check_sphinx():
    """Skip if sphinx is not installed."""
    pytest.importorskip("sphinx")


def test_docs_build(tmp_path: Path):
    """Verify that sphinx-build completes without errors."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            str(DOCS_DIR),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"sphinx-build failed (exit code {result.returncode}).\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )
