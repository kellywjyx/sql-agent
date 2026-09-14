import pytest


@pytest.fixture(autouse=True)
def isolated_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_TRACES", str(tmp_path / "traces.jsonl"))
