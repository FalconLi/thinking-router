"""E4: four live policies on a prespecified subset; needs uncontended CUDA capacity."""

import numpy as np
import pandas as pd

from .config import environment, load_config, parser, require_specification, run_location
from .evaluate import save_table
from .prepare_data import questions
from .runtime import execute, load_runtime_policies
from .stats import bootstrap_indices, interval
from .storage import append_jsonl, index_rows, read_json, read_jsonl, timestamp, write_json, writer_lock


def main():
    ap = parser(__doc__)
    ap.add_argument("--session", default="session-1", help="Use a new name for a new hardware/timing session")
    args = ap.parse_args()
    if not args.session.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Session must be a simple name")
    config = load_config(args.config)
    require_specification(config)
    policies = load_runtime_policies(config)
    from .backends import HFBackend

    backend = HFBackend(config)
    path = run_location(config, "results", "raw", f"e4_{args.session}.jsonl")
    import torch

    metadata_path = run_location(config, "results", "e4_sessions", f"{args.session}.json")
    metadata = {
        "run_id": config["run_id"],
        "session": args.session,
        "software": environment(),
        "gpu": torch.cuda.get_device_name(backend.model.device),
        "cuda": torch.version.cuda,
        "precision": config["precision"],
        "backend": "hf",
        "batch_size": 1,
    }
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise ValueError("Timing environment changed; choose a new --session name")
    else:
        write_json(metadata_path, {**metadata, "created_at": timestamp()})
    with writer_lock(path):
        existing = read_jsonl(path, recover=True)
        done = index_rows(existing, ("id", "method"))
        for row in questions(config, "warmup"):
            for policy in policies.values():
                execute(backend, policy, row["question"], row["id"], row["gold_number"])
        names = list(policies)
        for i, row in enumerate(questions(config, "latency")):
            for j in range(len(names)):
                name = names[(i + j) % len(names)]
                if (row["id"], name) in done:
                    continue
                result = execute(backend, policies[name], row["question"], row["id"], row["gold_number"])
                result["session"] = args.session
                result["recorded_at"] = timestamp()
                append_jsonl(path, result)
                print(
                    {"question": i + 1, "method": name, "seconds": result["end_to_end_seconds"]}, flush=True
                )
    frame = pd.DataFrame(read_jsonl(path))
    compact = frame.drop(
        columns=["raw_output", "generated_token_ids", "thinking_segment", "final_answer_segment"]
    )
    save_table(config, "e4_latency_per_question", compact)
    ids = [r["id"] for r in questions(config, "latency")]
    pivot = frame.pivot(index="id", columns="method", values="end_to_end_seconds").loc[ids]
    if pivot.isna().any().any():
        raise ValueError("Incomplete latency session")
    indices = bootstrap_indices(
        len(ids), config["statistics"]["bootstrap_samples"], config["statistics"]["bootstrap_seed"]
    )
    reference = pivot["always_thinking"].to_numpy()
    summary = []
    for method, group in frame.groupby("method"):
        times = pivot[method].to_numpy()
        speedups = reference[indices].mean(1) / times[indices].mean(1)
        low, high = interval(speedups)
        summary.append(
            {
                "method": method,
                "session": args.session,
                "questions": len(group),
                "mean_seconds": float(times.mean()),
                "median_seconds": float(np.median(times)),
                "mean_routing_ms": float(1000 * group.routing_seconds.mean()),
                "mean_generation_seconds": float(group.generation_seconds.mean()),
                "mean_tokens": float(group.token_count.mean()),
                "accuracy": float(group.correct.mean()),
                "tokens_per_second": float(group.token_count.sum() / group.end_to_end_seconds.sum()),
                "speedup": float(reference.mean() / times.mean()),
                "speedup_ci_low": low,
                "speedup_ci_high": high,
            }
        )
    save_table(config, "e4_latency_summary", summary)


if __name__ == "__main__":
    main()
