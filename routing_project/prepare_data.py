"""Unpack and verify the GSM8K questions and fixed experiment splits."""

import hashlib
import json
import tarfile
from pathlib import Path

from .config import load_config, location, parser
from .storage import align, atomic_write, file_hash, read_json, read_jsonl

INPUT_FILES = {
    *(f"questions/{split}.jsonl" for split in ("train", "validation", "test")),
    *(f"splits/{split}.json" for split in ("train", "validation", "test", "warmup", "latency", "subsets")),
}


def validate_manifest(config, manifest):
    dataset = {key: config["dataset"][key] for key in ("name", "config", "revision")}
    training = {key: config["training"][key] for key in ("sizes", "subset_seeds")}
    if (
        manifest["dataset"] != dataset
        or manifest["data"] != config["data"]
        or manifest["training"] != training
        or set(manifest["files"]) != INPUT_FILES
    ):
        raise ValueError("Dataset bundle does not match the experiment settings")


def verify_inputs(config):
    root = location(config, "data")
    manifest = read_json(root / "manifest.json")
    validate_manifest(config, manifest)
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Dataset input changed: {name}")
    return manifest


def prepare(config):
    archive_path = Path(config["_root"]) / "datasets/gsm8k.tar.gz"
    if file_hash(archive_path) != config["dataset"]["archive_sha256"]:
        raise ValueError("Dataset archive checksum mismatch")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        expected = INPUT_FILES | {"manifest.json"}
        if len(members) != len(expected) or {m.name for m in members} != expected:
            raise ValueError("Unexpected archive contents")
        if not all(member.isfile() for member in members):
            raise ValueError("Dataset archive must contain regular files only")
        content = {member.name: archive.extractfile(member).read() for member in members}
    manifest = json.loads(content["manifest.json"])
    validate_manifest(config, manifest)
    for name, expected in manifest["files"].items():
        if hashlib.sha256(content[name]).hexdigest() != expected:
            raise ValueError(f"Dataset member checksum mismatch: {name}")
    # Check every destination before writing, so a conflicting file is never replaced.
    for name, raw in content.items():
        path = location(config, "data", name)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Existing input differs from the bundle: {path}")
    for name, raw in content.items():
        path = location(config, "data", name)
        if not path.exists():
            atomic_write(path, lambda temporary, payload=raw: temporary.write_bytes(payload))
    verify_inputs(config)
    return {split: len(questions(config, split)) for split in ("train", "validation", "test")}


def questions(config, split):
    source = "train" if split == "warmup" else "test" if split == "latency" else split
    rows = read_jsonl(location(config, "data", "questions", f"{source}.jsonl"))
    if not rows:
        raise ValueError("Question files missing; run python -m routing_project.prepare_data first")
    ids = read_json(location(config, "data", "splits", f"{split}.json"))
    return align(rows, ids)


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args.config)
    print({"questions": prepare(config), "directory": str(location(config, "data"))})


if __name__ == "__main__":
    main()
