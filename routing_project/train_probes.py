"""Fit training-only preprocessing and validation-selected logistic probes."""

import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config, location, parser, require_specification
from .experiment_data import load_split
from .stats import auroc
from .storage import atomic_write, manifest_path, read_json, write_json


def probe_dir(config):
    return location(config, "artifacts", "probes", config["run_id"])


def make_pipeline_for(feature_type, c, iterations):
    preprocessing = (
        StandardScaler()
        if feature_type == "hidden"
        else TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_features=10000, sublinear_tf=True, norm="l2"
        )
    )
    return make_pipeline(
        preprocessing,
        LogisticRegression(C=c, solver="lbfgs", class_weight=None, max_iter=iterations, random_state=2026),
    )


def train(config):
    if manifest_path(config).exists():
        raise ValueError("Policies already frozen; refusing to retrain this run")
    train_data, validation = load_split(config, "train", paired=False), load_split(config, "validation")
    subsets = read_json(location(config, "data", "splits", "subsets.json"))
    id_index = {item: i for i, item in enumerate(train_data["ids"])}
    metadata = []
    for condition, ids in subsets.items():
        order = [id_index[item] for item in ids]
        y = train_data["a_nt"][order]
        for feature_type in ("hidden", "tfidf"):
            key = f"{feature_type}_{condition}"
            record = {
                "id": key,
                "condition": condition,
                "feature_type": feature_type,
                "training_ids": ids,
                "training_size": len(ids),
                "subset_seed": None if condition.endswith("full") else int(condition.split("_s")[1]),
                "config_hash": config["run_id"],
                "validation_label_count": len(validation["ids"]),
            }
            if len(np.unique(y)) < 2:
                record.update(status="single_class", validation_auroc=None, selected_c=None)
                metadata.append(record)
                continue
            x = train_data["hidden" if feature_type == "hidden" else "texts"][order]
            xv = validation["hidden" if feature_type == "hidden" else "texts"]
            choices = []
            best, best_auc = None, -np.inf
            for c in sorted(config["training"]["c_grid"]):
                iterations = config["training"]["max_iter"]
                while True:
                    pipeline = make_pipeline_for(feature_type, c, iterations)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        pipeline.fit(x, y)
                    if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        break
                    iterations *= 2
                    if iterations > 64000:
                        raise RuntimeError(f"Probe failed to converge: {key}, C={c}")
                score = auroc(validation["a_nt"], pipeline.predict_proba(xv)[:, 1])
                if not np.isfinite(score):
                    raise ValueError("Validation AUROC requires both correctness classes")
                choices.append({"c": c, "validation_auroc": score, "max_iter": iterations})
                if score > best_auc:
                    best, best_auc, selected_c = pipeline, score, c
            path = probe_dir(config) / f"{key}.joblib"
            atomic_write(path, lambda tmp, fitted=best: joblib.dump(fitted, tmp))
            record.update(
                status="fitted",
                validation_auroc=best_auc,
                selected_c=selected_c,
                c_at_grid_edge=selected_c
                in (min(config["training"]["c_grid"]), max(config["training"]["c_grid"])),
                train_auroc=auroc(y, best.predict_proba(x)[:, 1]),
                candidates=choices,
                pipeline=str(path.relative_to(config["_root"])),
            )
            metadata.append(record)
            write_json(probe_dir(config) / f"{key}.json", record)
            print({"probe": key, "validation_auroc": best_auc, "selected_c": selected_c}, flush=True)
    write_json(probe_dir(config) / "index.json", metadata)
    return metadata


def main():
    ap = parser(__doc__)
    args = ap.parse_args()
    config = load_config(args.config)
    require_specification(config)
    train(config)


if __name__ == "__main__":
    main()
