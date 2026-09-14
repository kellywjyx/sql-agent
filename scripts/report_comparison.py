"""Offline comparisons of frozen predictions; no model calls or dataset mutation.

Copied into each application's scripts directory for independent reproduction.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

from llm_evals import Metric, load_cases
from llm_evals.dataset import dataset_fingerprint, write_json
from llm_evals.runner import latest_run, paired_comparison

MODES = {"repo-sage": ["dense", "hybrid", "reranked", "agentic"],
         "sql-agent": ["baseline", "linking", "corrected"],
         "doc-extract": ["ocr", "vlm", "hybrid"]}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def per_case(project, cases, predictions, analysis):
    if project == "repo-sage":
        from repo_sage.evaluation import case_scores
        return [case_scores(case, pred) for case, pred in zip(cases, predictions)]
    if project == "doc-extract":
        from doc_extract.metrics import receipt_scores
        return [{**receipt_scores(case.expected, pred.get("output")),
                 **({"item_f1_v2": None} if "line_items" not in case.expected else {})}
                for case, pred in zip(cases, predictions)]
    wrong = {row["id"] for row in analysis if row.get("correct") is False}
    if len(wrong) != len(analysis):
        raise ValueError("BIRD error analysis is missing a correctness verdict")
    return [{"guarded_bird_ex": float(p["status"] == "completed" and c.id not in wrong)}
            for c, p in zip(cases, predictions)]


def create(root, project, split):
    target = root / "v2" / project
    cases = load_cases(target / ("bird" if project == "sql-agent" else "data") / f"{split}.jsonl")
    ids = [case.id for case in cases]
    dataset_hash = dataset_fingerprint(cases)
    modes = MODES[project]
    runs, values, rows = {}, {}, {}
    for mode in modes:
        folder = latest_run(target / "eval" / split / mode)
        report = read(folder / "metrics.json")
        predictions_file = folder / "predictions.jsonl"
        predictions = jsonl(predictions_file)
        analysis = jsonl(folder / "failure-analysis.jsonl")
        if report["dataset_hash"] != dataset_hash or ids != [p["id"] for p in predictions]:
            raise ValueError("Dataset content or ordered case IDs differ; comparison rejected")
        if report["count"] != len(cases) or report["identity"]["mode"] != "live":
            raise ValueError("Incomplete or non-live comparison input")
        if not report["metric_versions"] or any(not math.isfinite(v) for v in report["metrics"].values()):
            raise ValueError("Unversioned or nonfinite metrics")
        if runs and report["metric_versions"] != runs[modes[0]]["metric_versions"]:
            raise ValueError("Metric versions differ; comparison rejected")
        scored = per_case(project, cases, predictions, analysis)
        rows[mode] = [{"id": case.id, "scores": score, "status": pred["status"]}
                      for case, pred, score in zip(cases, predictions, scored)]
        values[mode] = scored
        metric_names = {"repo-sage": ["source_span_recall", "citation_evidence_precision_proxy", "required_fact_coverage_proxy", "refusal_accuracy"],
                        "doc-extract": ["annotated_field_f1_v2", "item_f1_v2", "total_exact_match"],
                        "sql-agent": ["guarded_bird_ex"]}[project]
        for metric in metric_names:
            available = [s[metric] for s in scored if s.get(metric) is not None]
            if not available or abs(sum(available) / len(available) - report["metrics"][metric]) > 1e-9:
                raise ValueError(f"Recorded aggregate disagrees with case scores: {mode}/{metric}")
        runs[mode] = {key: report[key] for key in ["run_id", "created_at", "count", "completed", "failed", "skipped", "status",
                     "metrics", "metric_versions", "confidence_intervals", "p50_seconds", "p95_seconds", "identity", "gate_failures"]}
        runs[mode]["predictions_sha256"] = hashlib.sha256(predictions_file.read_bytes()).hexdigest()
        runs[mode]["analysis_categories"] = dict(Counter(row.get("category", row.get("reason", "review_candidate")) for row in analysis))
        if project == "sql-agent":
            runs[mode]["execution_details"] = report.get("details", {})
        elif project == "doc-extract":
            geometries = [read(p) for p in (target / "geometry-eval").glob("*/metrics.json")]
            matching = [g for g in geometries if g["extraction_run_id"] == report["run_id"]
                        and g["predictions_sha256"] == runs[mode]["predictions_sha256"]]
            runs[mode]["geometry"] = {k: v for k, v in matching[-1].items() if k != "cases"} if matching else None
    pairs = {}
    for mode in modes[1:]:
        pairs[mode] = {}
        for name in metric_names:
            def average(cs, ps, metric=name):
                observations = [p["scores"][metric] for p in ps if p["scores"].get(metric) is not None]
                if not observations:
                    raise ValueError("No annotated observations")
                return sum(observations) / len(observations)
            pairs[mode][name] = paired_comparison(Metric(name, average, version=runs[mode]["metric_versions"][name]),
                                                 cases, rows[modes[0]], rows[mode])
    categories = {}
    for category in sorted({c.metadata.get("category", c.metadata.get("difficulty", "all")) for c in cases}):
        indices = [i for i, c in enumerate(cases) if c.metadata.get("category", c.metadata.get("difficulty", "all")) == category]
        categories[category] = {"count": len(indices), "modes": {}}
        for mode in modes:
            categories[category]["modes"][mode] = {}
            for name in metric_names:
                observed = [values[mode][i][name] for i in indices if values[mode][i].get(name) is not None]
                categories[category]["modes"][mode][name] = sum(observed) / len(observed) if observed else None
    summary = {"schema_version": 1, "project": project, "split": split, "dataset_hash": dataset_hash,
               "all_cases_attempted": True, "execution_gate_passed": all(r["status"] == "completed" for r in runs.values()),
               "baseline": modes[0], "runs": runs, "paired_comparisons": pairs, "by_category": categories,
               "local_api_charge_usd": 0, "hardware_electricity_cost_usd": None,
               "limitations": ["Comparisons summarize the same fixed cases, not independent benchmark replications.",
                               "Paired intervals use 300 deterministic resamples; no multiple-comparison correction.",
                               "Failed examples remain in denominators. A descriptive comparison does not pass a failed execution gate.",
                               "First/subsequent latency is not proof of cold/warm residency; see the separate model-residency probe."]}
    if project == "repo-sage":
        summary["limitations"].extend(["120 synthetic source-navigation questions, with 60 development and 60 final; not a representative code-assistant benchmark.",
                                       "Citation overlap and required-fact presence are proxies; source-rubric semantic review is separate."])
    elif project == "sql-agent":
        dataset_label = "100 BIRD training development cases" if split == "development" else "500-case BIRD SQLite Mini-Dev"
        summary["limitations"].append(f"Guarded {dataset_label} using the pinned upstream set-of-rows comparator; not a leaderboard result.")
    else:
        summary["limitations"].append("CORD receipt-field accuracy and highlight localization are separate measures; neither is calibrated confidence.")
    output = target / "comparisons" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    lines = [f"# {project}: {split} comparison", "", f"Dataset content identity: `{dataset_hash}`.", "",
             f"Cases per setting: {len(cases)}. Execution gate passed: **{summary['execution_gate_passed']}**.", "",
             "| Setting | " + " | ".join(metric_names) + " | Failed / skipped | p95 seconds |",
             "|---|" + "---:|" * (len(metric_names) + 2)]
    for mode, report in runs.items():
        lines.append(f"| {mode} | " + " | ".join(f"{report['metrics'][m]:.4f}" for m in metric_names)
                     + f" | {report['failed']} / {report['skipped']} | {report['p95_seconds']:.2f} |")
    lines.extend(["", "## Paired deltas against " + modes[0], "", "| Candidate | Metric | Delta | 95% interval |", "|---|---|---:|---|"])
    for mode, metrics in pairs.items():
        for name, value in metrics.items():
            lines.append(f"| {mode} | {name} | {value['delta']:+.4f} | [{value['low']:+.4f}, {value['high']:+.4f}] |")
    lines.extend(["", "## Run identities", "", *[f"- {mode}: `{r['run_id']}`; predictions SHA256 `{r['predictions_sha256']}`." for mode, r in runs.items()],
                  "", "## Limitations", "", *["- " + text for text in summary["limitations"]],
                  "- Local API charges: US$0. Hardware/electricity costs were not measured.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(target / f"{split}-comparison-latest.json", {"directory": output.name})
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--project", choices=list(MODES), default=Path(__file__).resolve().parents[1].name)
    parser.add_argument("--split", choices=["development", "final"], required=True)
    args = parser.parse_args()
    if args.project not in MODES:
        parser.error("--project is required outside an application repository")
    create(args.artifacts.resolve(), args.project, args.split)
