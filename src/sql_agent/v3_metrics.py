"""Bounded reference execution cached only in scoring, with failed denominators."""
from pathlib import Path
from llm_evals import Metric
from .execution import execute_sql
from .evaluation import equivalent


class Scoring:
    def __init__(self, cases):
        self.gold, self.errors = {}, {}
        self._scores = {}
        for case in cases:
            try:
                self.gold[case.id] = execute_sql(Path(case.context["database"]), case.expected["sql"], timeout=10)
            except (ValueError, TimeoutError) as error:
                self.errors[case.id] = str(error)

    def correct(self, case, prediction):
        cached = self._scores.get((case.id, id(prediction)))
        if cached and cached[0] is prediction:
            return cached[1]
        value = bool(case.id in self.gold and "rows" in prediction and "columns" in prediction and prediction.get("status") == "completed" and
                     equivalent(self.gold[case.id], prediction, case.expected.get("ordered", False)))
        # Keep the object alive to prevent id reuse; never trust model-supplied scores.
        self._scores[(case.id, id(prediction))] = (prediction, value)
        return value

    def metrics(self):
        return [Metric("execution_accuracy_v3", lambda cs, ps: sum(self.correct(c, p) for c, p in zip(cs, ps)) / len(cs), version="3-duplicate-order-preserving"),
            Metric("reference_coverage", lambda cs, ps: sum(c.id in self.gold for c in cs) / len(cs), version="3"),
            Metric("coverage", lambda cs, ps: sum(p.get("status") == "completed" for p in ps) / len(cs), version="3"),
            Metric("first_attempt_accuracy", lambda cs, ps: sum(self.correct(c, {**p.get("attempts", [{}])[0].get("result", {}), "status": "completed"}) for c, p in zip(cs, ps)) / len(cs), version="3"),
            Metric("correct_recovery_fraction", lambda cs, ps: sum(self.correct(c, p) and len(p.get("attempts", [])) > 1 and not self.correct(c, {**p["attempts"][0].get("result", {}), "status": "completed"}) for c, p in zip(cs, ps)) / len(cs), version="3"),
            Metric("timeout_fraction", lambda cs, ps: sum(any(a.get("category") == "timeout" for a in p.get("attempts", [])) for p in ps) / len(cs), version="3")]
