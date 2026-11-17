"""Choose on validation, freeze all artifacts, then permit test execution."""

import math
from pathlib import Path

import joblib
import numpy as np

from .config import load_config, location, parser, require_specification
from .experiment_data import load_split
from .policies import choose_policy
from .storage import manifest_path, read_json, write_json, write_manifest
from .train_probes import probe_dir


def policy_dir(config):
    return location(config, "artifacts", "policies", config["run_id"])


def scores_for(config, record, data):
    method = record["feature_type"]
    if method == "length":
        return -data["length"].astype(float)
    if method == "random" or record.get("status") == "single_class":
        return np.zeros(len(data["ids"]))
    pipeline = joblib.load(Path(config["_root"]) / record["pipeline"])
    return pipeline.predict_proba(data["hidden"] if method == "hidden" else data["texts"])[:, 1]


def json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    return value


def select(config):
    if manifest_path(config).exists():
        raise ValueError("This run is already frozen; refusing to replace policies")
    data = load_split(config, "validation")
    probe_index = read_json(probe_dir(config) / "index.json")
    expected = len(read_json(location(config, "data", "splits", "subsets.json"))) * 2
    if len(probe_index) != expected:
        raise ValueError("Freeze requires every prespecified training condition")
    records = probe_index + [
        {"id": name, "feature_type": name, "training_size": 0, "subset_seed": None}
        for name in ("length", "random")
    ]
    selected, all_candidates = [], {}
    for record in records:
        scores = scores_for(config, record, data)
        for tolerance in config["policy"]["tolerances"]:
            choice, candidates = choose_policy(data, scores, record["feature_type"], tolerance)
            if record.get("status") == "single_class":
                choice = next(row for row in candidates if row["kind"] == "always_thinking")
            item = {
                **record,
                **choice,
                "policy_id": f"{record['id']}_tol{tolerance:.2f}",
                "tolerance": tolerance,
                "primary": tolerance == config["policy"]["primary_tolerance"],
            }
            selected.append(item)
            all_candidates[record["id"]] = candidates
    out = policy_dir(config)
    write_json(out / "selected.json", json_finite(selected))
    write_json(out / "candidates.json", json_finite(all_candidates))
    files = list(probe_dir(config).glob("*")) + [
        out / "selected.json",
        out / "candidates.json",
        Path(config["_path"]),
    ]
    files += list(location(config, "data", "splits").glob("*.json"))
    files += list(location(config, "data", "questions").glob("*.jsonl"))
    files += list((Path(config["_root"]) / "routing_project").glob("*.py"))
    # Freeze training/validation inputs too: changing labels/features invalidates the run.
    for split in ("train", "validation"):
        files += list(location(config, "data", "generations", config["run_id"]).glob(f"{split}_*.jsonl"))
        files += list(location(config, "data", "features", config["run_id"]).glob(f"{split}.*"))
    write_manifest(config, files)
    print({"selected_policies": len(selected), "manifest": str(manifest_path(config))})
    return selected


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args.config)
    require_specification(config)
    select(config)


if __name__ == "__main__":
    main()
