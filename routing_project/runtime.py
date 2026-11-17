"""Shared actual routed generation, including repeated prompt processing."""

import time
from pathlib import Path

import joblib

from .grading import grade_response
from .modeling import build_prompt_ids, extract_batch, seed_for
from .policies import Policy
from .select_policies import policy_dir
from .storage import read_json, verify_manifest


def load_runtime_policies(config):
    verify_manifest(config)
    selected = read_json(policy_dir(config) / "selected.json")
    policies = {
        "always_thinking": Policy("always_thinking", "always_thinking"),
        "never_thinking": Policy("never_thinking", "never_thinking"),
    }
    for name in ("tfidf", "hidden"):
        record = next(
            r
            for r in selected
            if r["feature_type"] == name
            and r["primary"]
            and r["training_size"] == config["data"]["train_size"]
        )
        pipeline = joblib.load(Path(config["_root"]) / record["pipeline"]) if record.get("pipeline") else None
        policies[name] = Policy(name, record["kind"], record.get("threshold"), pipeline=pipeline)
    return policies


def execute(backend, policy, question, question_id, gold=None):
    import torch

    torch.cuda.synchronize(backend.model.device)
    start = time.perf_counter()
    score = None
    if policy.kind == "threshold":
        if policy.method == "hidden":
            feature = extract_batch(backend.model, backend.tokenizer, [question])[0]
            score = policy.score(question, feature)
        else:
            score = policy.score(question)
    mode = policy.choose_mode(score)
    torch.cuda.synchronize(backend.model.device)
    route_end = time.perf_counter()
    prompt = build_prompt_ids(backend.tokenizer, question, mode)
    output = backend.generate_one(
        prompt, mode, seed_for(question_id, mode, backend.config["generation"]["base_seed"])
    )
    text = backend.tokenizer.decode(output["token_ids"], skip_special_tokens=False)
    torch.cuda.synchronize(backend.model.device)
    end = time.perf_counter()
    row = {
        "id": question_id,
        "method": policy.method,
        "mode": mode,
        "score": score,
        "end_to_end_seconds": end - start,
        "routing_seconds": route_end - start,
        "generation_seconds": end - route_end,
        "raw_output": text,
    }
    if gold is not None:
        row.update(
            grade_response(
                output["token_ids"],
                backend.tokenizer,
                gold,
                backend.stop_ids,
                backend.config["generation"]["max_new_tokens"],
            )
        )
    return row
