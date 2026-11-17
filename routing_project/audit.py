"""Check prepared splits or complete formal artifacts and save a machine-readable audit."""

from .config import load_config, location, parser, run_location
from .experiment_data import load_split
from .prepare_data import questions, verify_inputs
from .storage import read_json, timestamp, verify_manifest, write_json


def audit(config, formal=False):
    verify_inputs(config)
    split_ids = {
        split: read_json(location(config, "data", "splits", f"{split}.json"))
        for split in ("train", "validation", "test", "warmup", "latency")
    }
    for name, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {name}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(split_ids[a]) & set(split_ids[b]):
            raise ValueError(f"Overlapping splits: {a}/{b}")
    for child, parent in (("warmup", "train"), ("latency", "test")):
        if not set(split_ids[child]) <= set(split_ids[parent]):
            raise ValueError(f"Invalid subset: {child}")
    for split in ("train", "validation", "test"):
        rows = questions(config, split)
        expected = config["data"][f"{split}_size"]
        if len(rows) != expected:
            raise ValueError(f"Wrong question count for {split}")
    report = {
        "run_id": config["run_id"],
        "checked_at": timestamp(),
        "stage": "formal" if formal else "prepared",
        "status": "passed",
        "counts": {name: len(ids) for name, ids in split_ids.items()},
        "dataset_revision": config["dataset"]["revision"],
    }
    if formal:
        verify_manifest(config)
        total_responses, total_features = 0, 0
        for split in ("train", "validation", "test"):
            data = load_split(config, split, paired=split != "train")
            total_features += len(data["ids"])
            total_responses += len(data["ids"]) * (1 if split == "train" else 2)
        report.update(response_records=total_responses, feature_vectors=total_features)
    path = run_location(config, "results", "checks", f"audit_{report['stage']}.json")
    write_json(path, report)
    print({"report": str(path), "status": "passed", "stage": report["stage"]})
    return report


def main():
    ap = parser(__doc__)
    ap.add_argument("--formal", action="store_true")
    args = ap.parse_args()
    audit(load_config(args.config), args.formal)


if __name__ == "__main__":
    main()
