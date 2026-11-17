"""Generate and resume compatible per-question caches."""

from .config import load_config, location, parser, require_specification
from .grading import grade_response
from .modeling import build_prompt_ids, seed_for
from .prepare_data import questions
from .storage import append_jsonl, index_rows, read_jsonl, verify_manifest, writer_lock


def cache_path(config, split, mode):
    return location(config, "data", "generations", config["run_id"], f"{split}_{mode}.jsonl")


def generate_split(config, split, modes, backend=None, rows=None):
    if split == "test":
        verify_manifest(config)
    rows = questions(config, split) if rows is None else rows
    for mode in modes:
        path = cache_path(config, split, mode)
        with writer_lock(path):
            existing = read_jsonl(path, recover=True)
            index_rows(existing)
            if any(row["config_hash"] != config["run_id"] or row["mode"] != mode for row in existing):
                raise ValueError("Incompatible generation cache")
            completed = {row["id"] for row in existing}
            pending = [row for row in rows if row["id"] not in completed]
            print({"output": str(path), "completed": len(completed), "remaining": len(pending)}, flush=True)
            if pending and backend is None:
                from .backends import make_backend

                backend = make_backend(config)
            batch_size = 1
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                requests = [
                    {
                        "prompt_ids": build_prompt_ids(backend.tokenizer, row["question"], mode),
                        "mode": mode,
                        "seed": seed_for(row["id"], mode, config["generation"]["base_seed"]),
                    }
                    for row in batch
                ]
                outputs = backend.generate_many(requests)
                if len(outputs) != len(batch):
                    raise ValueError("Backend returned wrong number of outputs")
                for row, request, output in zip(batch, requests, outputs, strict=True):
                    graded = grade_response(
                        output["token_ids"],
                        backend.tokenizer,
                        row["gold_number"],
                        backend.stop_ids,
                        config["generation"]["max_new_tokens"],
                    )
                    record = {
                        "id": row["id"],
                        "split": split,
                        "mode": mode,
                        "config_hash": config["run_id"],
                        "backend": config["backend"],
                        "seed": request["seed"],
                        **{k: v for k, v in output.items() if k != "token_ids"},
                        **graded,
                    }
                    append_jsonl(path, record)
                print(
                    {"mode": mode, "completed": len(completed) + start + len(batch), "total": len(rows)},
                    flush=True,
                )
    return backend


def main():
    ap = parser(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--modes", nargs="+", required=True, choices=["nonthinking", "thinking"])
    args = ap.parse_args()
    config = load_config(args.config)
    require_specification(config)
    generate_split(config, args.split, args.modes)


if __name__ == "__main__":
    main()
