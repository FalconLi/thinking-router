"""Reproduce E1–E3 from the frozen cache without loading a language model."""

import numpy as np
import pandas as pd

from .config import load_config, parser, require_specification, run_location
from .experiment_data import load_split
from .policies import fixed_fraction_weights, nonthinking_weights
from .select_policies import policy_dir, scores_for
from .stats import auroc, bootstrap_indices, bootstrap_metrics, interval, metrics, with_intervals
from .storage import atomic_write, read_json, verify_manifest, write_json


def save_table(config, name, rows):
    path = run_location(config, "results", "curves" if "curves" in name else "tables", f"{name}.csv")
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.insert(0, "run_id", config["run_id"])
    atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))
    return path


def quality_metrics(data, weights):
    failures = {mode: np.array([r["parse_status"] != "parsed" for r in data[mode]]) for mode in ("nt", "t")}
    truncated = {mode: np.array([r["truncated"] for r in data[mode]]) for mode in ("nt", "t")}
    return {
        "parse_failure_fraction": float(np.mean(weights * failures["nt"] + (1 - weights) * failures["t"])),
        "truncation_fraction": float(np.mean(weights * truncated["nt"] + (1 - weights) * truncated["t"])),
    }


def evaluate(config, experiments=("e1", "e2", "e3")):
    verify_manifest(config)
    data = load_split(config, "test")
    n = len(data["ids"])
    indices = bootstrap_indices(
        n, config["statistics"]["bootstrap_samples"], config["statistics"]["bootstrap_seed"]
    )
    selected = read_json(policy_dir(config) / "selected.json")
    candidate_map = read_json(policy_dir(config) / "candidates.json")
    point_rows, distributions, score_cache, fixed, curves = [], {}, {}, [], []
    full_size = config["data"]["train_size"]
    endpoints = []
    for method, weights in (
        ("always_thinking", np.zeros(n)),
        ("never_thinking", np.ones(n)),
        ("oracle", data["a_nt"].copy()),
    ):
        point = {**metrics(data, weights), **quality_metrics(data, weights)}
        boot = bootstrap_metrics(data, weights, indices)
        row = {
            "method": method,
            "training_size": 0,
            "primary": True,
            "tolerance": 0.01,
            **with_intervals(point, boot),
        }
        endpoints.append(row)
        point_rows.append(row)
        if method != "oracle":
            records = data["t"] if method == "always_thinking" else data["nt"]
            row.update(
                median_tokens=float(np.median([r["token_count"] for r in records])),
                parse_failure_fraction=float(np.mean([r["parse_status"] != "parsed" for r in records])),
                truncation_fraction=float(np.mean([r["truncated"] for r in records])),
            )
    paired = [
        {
            "nt_correct": nt,
            "t_correct": t,
            "count": int(((data["a_nt"] == nt) & (data["a_t"] == t)).sum()),
            "percentage": float(100 * ((data["a_nt"] == nt) & (data["a_t"] == t)).mean()),
        }
        for nt in (0, 1)
        for t in (0, 1)
    ]
    if "e1" in experiments:
        save_table(config, "e1_endpoints", endpoints)
        save_table(config, "e1_paired_outcomes", paired)
    learning = []
    for policy in selected:
        method, key = policy["feature_type"], policy["id"]
        primary = policy["primary"]
        headline = policy["training_size"] in (0, full_size)
        if key not in score_cache:
            score_cache[key] = scores_for(config, policy, data)
        scores = score_cache[key]
        weights = nonthinking_weights(policy, scores)
        point = {**metrics(data, weights), **quality_metrics(data, weights)}
        has_score = method in {"hidden", "tfidf", "length"} and policy.get("status") != "single_class"
        if has_score:
            point["auroc"] = auroc(data["a_nt"], scores)
        row = {
            "method": method,
            "policy_id": policy["policy_id"],
            "training_size": policy["training_size"],
            "subset_seed": policy["subset_seed"],
            "primary": primary,
            "tolerance": policy["tolerance"],
            "selected_on": "validation",
            **point,
        }
        if headline:
            boot = bootstrap_metrics(data, weights, indices, scores if has_score else None)
            row = with_intervals(row, boot)
            point_rows.append(row)
            if primary:
                distributions[method] = boot
                for candidate in candidate_map[key]:
                    curves.append(
                        {
                            "method": method,
                            "training_size": policy["training_size"],
                            "subset_seed": policy["subset_seed"],
                            "kind": candidate["kind"],
                            "threshold": candidate.get("threshold"),
                            "probability": candidate.get("probability"),
                            **metrics(data, nonthinking_weights(candidate, scores)),
                        }
                    )
                if has_score:
                    for fraction in (0.25, 0.5, 0.75):
                        w = fixed_fraction_weights(data["ids"], scores, fraction)
                        fixed.append(
                            {
                                "method": method,
                                "target_thinking_fraction": fraction,
                                "analysis_type": "test_ranking_not_deployable",
                                **with_intervals(metrics(data, w), bootstrap_metrics(data, w, indices)),
                            }
                        )
        if primary and method in {"hidden", "tfidf"}:
            learning.append(row)
    if "e2" in experiments:
        primary_rows = [row for row in point_rows if row["primary"]]
        save_table(config, "e2_selected_policies", primary_rows)
        save_table(config, "e2_tolerance_sensitivity", point_rows)
        save_table(config, "e2_fixed_fraction", fixed)
        save_table(config, "e2_routing_curves", curves)
        auc_rows = [
            {k: row[k] for k in row if k in {"method", "training_size"} or k.startswith("auroc")}
            for row in primary_rows
            if "auroc" in row
        ]
        save_table(config, "e2_score_auroc", auc_rows)
        differences = []
        point_by_method = {r["method"]: r for r in primary_rows}
        for baseline in ("tfidf", "length"):
            for metric in ("auroc", "accuracy", "token_saving_pct"):
                if metric not in distributions.get("hidden", {}) or metric not in distributions.get(
                    baseline, {}
                ):
                    continue
                values = distributions["hidden"][metric] - distributions[baseline][metric]
                low, high = interval(values)
                differences.append(
                    {
                        "comparison": f"hidden_minus_{baseline}",
                        "metric": metric,
                        "difference": point_by_method["hidden"][metric] - point_by_method[baseline][metric],
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_valid": int(np.isfinite(values).sum()),
                        "excludes_zero": bool(low > 0 or high < 0),
                    }
                )
        save_table(config, "e2_paired_differences", differences)
        # Save aligned per-question outcomes and routing decisions for further analysis.
        question_rows = []
        main_policies = [p for p in selected if p["primary"] and p["training_size"] in (0, full_size)]
        decisions = {p["feature_type"]: nonthinking_weights(p, score_cache[p["id"]]) for p in main_policies}
        for i, item in enumerate(data["ids"]):
            row = {
                "id": item,
                "nt_correct": data["a_nt"][i],
                "t_correct": data["a_t"][i],
                "nt_tokens": data["g_nt"][i],
                "t_tokens": data["g_t"][i],
            }
            for p in main_policies:
                name = p["feature_type"]
                row[f"{name}_score"] = score_cache[p["id"]][i]
                row[f"{name}_nonthinking_weight"] = decisions[name][i]
            question_rows.append(row)
        save_table(config, "e2_per_question", question_rows)
    if "e3" in experiments:
        save_table(config, "e3_learning_curve_runs", learning)
        frame = pd.DataFrame(learning)
        summaries = []
        for (method, size), group in frame.groupby(["method", "training_size"]):
            row = {
                "method": method,
                "training_size": size,
                "subset_count": len(group),
                "validation_labels": config["data"]["validation_size"],
                "uncertainty_type": "test_bootstrap" if size == full_size else "subset_standard_deviation",
            }
            for metric in ("auroc", "accuracy", "token_saving_pct"):
                row[f"{metric}_mean"] = float(group[metric].mean()) if metric in group else float("nan")
                row[f"{metric}_std"] = (
                    float(group[metric].std()) if len(group) > 1 and metric in group else float("nan")
                )
                if size == full_size:
                    for suffix in ("ci_low", "ci_high"):
                        row[f"{metric}_{suffix}"] = group.iloc[0].get(f"{metric}_{suffix}", float("nan"))
            summaries.append(row)
        save_table(config, "e3_learning_curve_summary", summaries)
    write_json(
        run_location(config, "results", "evaluation_manifest.json"),
        {
            "run_id": config["run_id"],
            "experiments": list(experiments),
            "test_questions": n,
            "bootstrap": config["statistics"],
            "policy_selection": "validation_only",
            "uncertainty_scope": "fixed fitted policies; cached response question sampling only",
        },
    )
    print({"results": str(run_location(config, "results")), "experiments": list(experiments), "questions": n})


def main():
    ap = parser(__doc__)
    ap.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3"], choices=["e1", "e2", "e3"])
    args = ap.parse_args()
    config = load_config(args.config)
    require_specification(config)
    evaluate(config, args.experiments)


if __name__ == "__main__":
    main()
