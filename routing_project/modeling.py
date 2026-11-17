"""Tokenizer helpers and shared HF feature extraction; no eager torch import."""

import hashlib

import numpy as np

PROMPT = "Solve the following problem. Put the final numeric answer in \\boxed{{...}}.\n\n{question}"


def seed_for(question_id, mode, base_seed=2026):
    digest = hashlib.sha256(f"{base_seed}|{question_id}|{mode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def load_tokenizer(config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config["model"]["name"], revision=config["model"]["revision"])


def build_prompt_ids(tokenizer, question, mode):
    if mode not in {"thinking", "nonthinking"}:
        raise ValueError(mode)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(question=question)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=mode == "thinking",
    )


def common_prefix_ids(tokenizer, question):
    first = build_prompt_ids(tokenizer, question, "thinking")
    second = build_prompt_ids(tokenizer, question, "nonthinking")
    end = 0
    for a, b in zip(first, second, strict=False):
        if a != b:
            break
        end += 1
    prefix = first[:end]
    if not prefix or not tokenizer.decode(prefix).endswith("<|im_start|>assistant\n"):
        raise ValueError("Unexpected chat-template common prefix; inspect the pinned tokenizer")
    return prefix


def tokenizer_hash(tokenizer):
    from .config import canonical_hash

    return canonical_hash({"vocab": tokenizer.get_vocab(), "template": tokenizer.chat_template})


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM

    if not config["device"].startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Model execution requires an allocated CUDA device")
    precision = config["precision"]
    if precision == "auto":
        precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"], torch_dtype=getattr(torch, precision)
    )
    model = model.to(config["device"]).eval()
    model.requires_grad_(False)
    return model


def extract_batch(model, tokenizer, question_texts):
    import torch

    prefixes = [common_prefix_ids(tokenizer, question) for question in question_texts]
    max_length = max(map(len, prefixes))
    ids = torch.full(
        (len(prefixes), max_length), tokenizer.pad_token_id, dtype=torch.long, device=model.device
    )
    mask = torch.zeros_like(ids)
    for i, prefix in enumerate(prefixes):
        ids[i, : len(prefix)] = torch.tensor(prefix, device=model.device)
        mask[i, : len(prefix)] = 1
    with torch.inference_mode():
        output = model.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    indices = torch.arange(len(prefixes), device=model.device)
    last = mask.sum(1) - 1
    vectors = output.last_hidden_state[indices, last].float().cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite feature vector")
    return vectors


def verify_feature_equivalence(model, tokenizer, texts):
    batch = extract_batch(model, tokenizer, texts)
    singles = np.concatenate([extract_batch(model, tokenizer, [text]) for text in texts])
    cosines = (batch * singles).sum(1) / (np.linalg.norm(batch, axis=1) * np.linalg.norm(singles, axis=1))
    result = {
        "min_cosine": float(cosines.min()),
        "max_abs_difference": float(abs(batch - singles).max()),
        "questions": len(texts),
    }
    if result["min_cosine"] < 0.999:
        raise ValueError(f"Batched feature mismatch: {result}")
    return result
