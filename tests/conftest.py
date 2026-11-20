import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def lab_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_HOME", str(tmp_path))
    return tmp_path
