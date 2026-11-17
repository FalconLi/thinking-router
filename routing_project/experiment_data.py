"""Strictly aligned cached responses and features shared by E1–E3."""

import numpy as np

from .config import location
from .prepare_data import questions
from .run_modes import cache_path
from .storage import align, atomic_write, read_json, read_jsonl, write_json


def feature_path(config, split):
    return location(config, "data", "features", config["run_id"], f"{split}.npz")


def save_features(config, split, ids, vectors, lengths, metadata):
    path = feature_path(config, split)

    def writer(tmp):
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                ids=np.asarray(ids),
                vectors=np.asarray(vectors, dtype=np.float32),
                lengths=np.asarray(lengths),
            )

    atomic_write(path, writer)
    write_json(path.with_suffix(".json"), {"run_id": config["run_id"], **metadata})


def load_features(config, split, ids):
    path = feature_path(config, split)
    metadata = read_json(path.with_suffix(".json"))
    if metadata["run_id"] != config["run_id"]:
        raise ValueError("Feature configuration mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        stored_ids = arrays["ids"].tolist()
        if len(stored_ids) != len(set(stored_ids)):
            raise ValueError("Duplicate feature IDs")
        indices = {item: i for i, item in enumerate(stored_ids)}
        order = [indices[item] for item in ids]
        vectors, lengths = arrays["vectors"][order], arrays["lengths"][order]
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite features")
    return vectors, lengths


def load_responses(config, split, mode, ids):
    rows = read_jsonl(cache_path(config, split, mode))
    if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
        raise ValueError(f"Incomplete or extra {split}/{mode} cache: {len(rows)} vs {len(ids)}")
    rows = align(rows, ids)
    if any(row["config_hash"] != config["run_id"] or row["mode"] != mode for row in rows):
        raise ValueError("Generation configuration mismatch")
    return rows


def load_split(config, split, paired=True):
    rows = questions(config, split)
    ids = [row["id"] for row in rows]
    vectors, lengths = load_features(config, split, ids)
    nonthinking = load_responses(config, split, "nonthinking", ids)
    result = {
        "ids": np.asarray(ids),
        "texts": np.asarray([row["question"] for row in rows]),
        "hidden": vectors,
        "length": lengths,
        "nt": nonthinking,
        "a_nt": np.asarray([row["correct"] for row in nonthinking], dtype=float),
        "g_nt": np.asarray([row["token_count"] for row in nonthinking], dtype=float),
    }
    if paired:
        thinking = load_responses(config, split, "thinking", ids)
        result.update(
            t=thinking,
            a_t=np.asarray([r["correct"] for r in thinking], dtype=float),
            g_t=np.asarray([r["token_count"] for r in thinking], dtype=float),
        )
    return result
