"""One threshold convention shared by cached evaluation and live execution."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Policy:
    method: str
    kind: str
    threshold: float | None = None
    probability: float | None = None
    pipeline: object = None

    def score(self, question, optional_feature=None):
        if self.method == "hidden":
            return float(self.pipeline.predict_proba(np.asarray(optional_feature).reshape(1, -1))[0, 1])
        if self.method == "tfidf":
            return float(self.pipeline.predict_proba([question])[0, 1])
        if self.method == "length":
            return -float(optional_feature)
        return 0.0

    def choose_mode(self, score):
        if self.kind == "always_thinking":
            return "thinking"
        if self.kind == "never_thinking":
            return "nonthinking"
        if self.kind == "random":
            raise ValueError("Random live routing requires an explicit RNG; not an E4 policy")
        return "nonthinking" if score >= self.threshold else "thinking"


def nonthinking_weights(policy, scores):
    kind = policy["kind"]
    n = len(scores)
    if kind == "always_thinking":
        return np.zeros(n)
    if kind == "never_thinking":
        return np.ones(n)
    if kind == "random":
        return np.full(n, 1 - policy["probability"])
    return (np.asarray(scores) >= policy["threshold"]).astype(float)


def candidates(scores, method):
    if method == "random":
        return [{"kind": "random", "probability": q / 100} for q in range(101)]
    distinct = np.unique(scores)
    thresholds = distinct[:-1] + np.diff(distinct) / 2
    return [
        {"kind": "never_thinking"},
        *[{"kind": "threshold", "threshold": float(t)} for t in thresholds],
        {"kind": "always_thinking"},
    ]


def choose_policy(data, scores, method, tolerance):
    from .stats import metrics

    evaluated = [
        {**candidate, "validation": metrics(data, nonthinking_weights(candidate, scores))}
        for candidate in candidates(scores, method)
    ]
    eligible = [
        row for row in evaluated if row["validation"]["accuracy"] >= data["a_t"].mean() - tolerance - 1e-12
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["validation"]["mean_tokens"],
            -row["validation"]["accuracy"],
            row["validation"]["thinking_fraction"],
        ),
    )
    return selected, evaluated


def fixed_fraction_weights(ids, scores, fraction):
    order = np.lexsort((np.asarray(ids), np.asarray(scores)))
    weights = np.ones(len(ids))
    weights[order[: round(fraction * len(ids))]] = 0
    return weights
