"""Atomic artifacts, strict ID joins, and crash-safe single-writer caches."""

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def timestamp():
    return datetime.now(UTC).isoformat()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_text(path, text):
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path, recover=False):
    path = Path(path)
    if not path.exists():
        return []
    rows, good_end = [], 0
    raw = path.read_bytes()
    for index, line in enumerate(raw.splitlines(keepends=True)):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if recover and good_end + len(line) == len(raw) and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(good_end)
                break
            raise ValueError(f"Corrupt JSONL row {index + 1}: {path}") from None
        rows.append(row)
        good_end += len(line)
    if recover and rows and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def writer_lock(path):
    path = Path(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another writer is active: {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def index_rows(rows, keys=("id",)):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate record: {key}")
        result[key] = row
    return result


def align(rows, ids):
    indexed = index_rows(rows)
    return [indexed[(item,)] for item in ids]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(config):
    from .config import location

    return location(config, "artifacts", "policies", config["run_id"], "freeze_manifest.json")


def write_manifest(config, files):
    root = Path(config["_root"])
    if manifest_path(config).exists():
        raise ValueError("Policies already frozen; use a new configuration for a new experiment")
    write_json(
        manifest_path(config),
        {
            "run_id": config["run_id"],
            "created_at": timestamp(),
            "files": {str(Path(p).relative_to(root)): file_hash(p) for p in sorted(files)},
        },
    )


def verify_manifest(config):
    manifest = read_json(manifest_path(config))
    if manifest["run_id"] != config["run_id"] or not manifest["files"]:
        raise ValueError("Wrong or empty freeze manifest")
    root = Path(config["_root"]).resolve()
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return manifest
