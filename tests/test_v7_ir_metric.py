import pytest

from sql_agent.v7_evaluation import V7Scoring


def _scorer():
    # Bypass __init__, which executes reference SQL against real databases.
    return V7Scoring.__new__(V7Scoring)


def test_ir_structural_accuracy_is_not_applicable_without_semantic_intent():
    with pytest.raises(ValueError, match="not applicable"):
        _scorer().ir_structural_accuracy([], [])


def test_pipelines_without_semantic_intent_can_exclude_the_metric():
    assert "ir_structural_accuracy" not in [metric.name for metric in _scorer().metrics(include_ir=False)]
    assert "ir_structural_accuracy" in [metric.name for metric in _scorer().metrics()]
