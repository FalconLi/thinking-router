"""Save ID-aligned final-normalized decoder features for requested splits."""

import numpy as np

from .config import load_config, parser, require_specification
from .experiment_data import feature_path, load_features, save_features
from .modeling import extract_batch, load_model, load_tokenizer, tokenizer_hash, verify_feature_equivalence
from .prepare_data import questions
from .storage import verify_manifest


def main():
    ap = parser(__doc__)
    ap.add_argument("--splits", nargs="+", required=True, choices=["train", "validation", "test"])
    args = ap.parse_args()
    config = load_config(args.config)
    require_specification(config)
    if "test" in args.splits:
        verify_manifest(config)
    tokenizer, model = None, None
    for split in args.splits:
        rows = questions(config, split)
        ids = [row["id"] for row in rows]
        if feature_path(config, split).exists():
            load_features(config, split, ids)
            print({"split": split, "completed": len(ids), "remaining": 0})
            continue
        if model is None:
            tokenizer, model = load_tokenizer(config), load_model(config)
            check = verify_feature_equivalence(
                model, tokenizer, [row["question"] for row in questions(config, "train")[:8]]
            )
        vectors = []
        batch_size = config["features"]["batch_size"]
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            features = extract_batch(
                model,
                tokenizer,
                [row["question"] for row in batch],
            )
            vectors.append(features)
        lengths = [len(tokenizer.encode(row["question"], add_special_tokens=False)) for row in rows]
        save_features(
            config,
            split,
            ids,
            np.concatenate(vectors),
            lengths,
            {
                "model_revision": config["model"]["revision"],
                "tokenizer_hash": tokenizer_hash(tokenizer),
                "position": "last_token_of_common_assistant_header",
                "layer": "final_norm",
                "padding_side": "right",
                "equivalence": check,
            },
        )
        print({"split": split, "completed": len(ids), "output": str(feature_path(config, split))})


if __name__ == "__main__":
    main()
