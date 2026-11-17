"""Experiment settings, paths, and reproducible run identifiers."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import yaml


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def run_id(config):
    settings = {
        k: v for k, v in config.items() if not k.startswith("_") and k not in {"paths", "device", "run_id"}
    }
    return "run-" + canonical_hash(settings)[:16]


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    require_specification(config)
    sizes = config["training"]["sizes"]
    if sorted(set(sizes)) != sizes or min(sizes) <= 0 or max(sizes) != config["data"]["train_size"]:
        raise ValueError("Training sizes must increase uniquely to the full training-pool size")
    if not 0 < config["data"]["warmup_size"] <= config["data"]["train_size"]:
        raise ValueError("Invalid warm-up size")
    if not 0 < config["data"]["latency_size"] <= config["data"]["test_size"]:
        raise ValueError("Invalid latency subset size")
    if config["policy"]["primary_tolerance"] not in config["policy"]["tolerances"]:
        raise ValueError("Primary tolerance must be in the prespecified tolerance list")
    if config["generation"]["max_new_tokens"] <= 0 or config["statistics"]["bootstrap_samples"] <= 0:
        raise ValueError("Token cap and bootstrap sample count must be positive")
    root = next(parent for parent in path.parents if (parent / "pyproject.toml").exists())
    config["_root"] = str(root)
    config["_path"] = str(path)
    config["run_id"] = run_id(config)
    return config


def location(config, kind, *parts):
    return Path(config["_root"]) / config["paths"][kind] / Path(*parts)


def run_location(config, kind, *parts):
    return location(config, kind, config["run_id"], *parts)


def environment():
    versions = {}
    for name in [
        "numpy",
        "scikit-learn",
        "transformers",
        "torch",
        "huggingface-hub",
        "pandas",
        "scipy",
        "joblib",
        "tokenizers",
        "safetensors",
        "matplotlib",
        "PyYAML",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def require_specification(config):
    if config.get("schema_version") != 1 or config.get("backend") != "hf":
        raise ValueError("The experiment requires schema version 1 and the HF backend")
    for item in (config["model"], config["dataset"]):
        if not re.fullmatch(r"[a-f0-9]{40}", item["revision"]):
            raise ValueError("Experiments require immutable model and dataset commit SHAs")


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default="configs/experiment.yaml")
    return result
