"""Paired question bootstrap; no refitting or policy reselection."""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def auroc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def gain_over_random(accuracy, cost, nt_accuracy, t_accuracy, nt_cost, t_cost):
    denominator = t_cost - nt_cost
    if abs(denominator) < 1e-12:
        return float("nan")
    q = (cost - nt_cost) / denominator
    if q < -1e-10 or q > 1 + 1e-10:
        return float("nan")  # no feasible equal-cost random policy; do not extrapolate
    return 100 * (accuracy - ((1 - q) * nt_accuracy + q * t_accuracy))


def outcomes(data, weights):
    return (
        weights * data["a_nt"] + (1 - weights) * data["a_t"],
        weights * data["g_nt"] + (1 - weights) * data["g_t"],
    )


def metrics(data, weights):
    correct, tokens = outcomes(data, weights)
    a, cost = float(correct.mean()), float(tokens.mean())
    t_cost, nt_cost = float(data["g_t"].mean()), float(data["g_nt"].mean())
    t_acc, nt_acc = float(data["a_t"].mean()), float(data["a_nt"].mean())
    return {
        "accuracy": a,
        "mean_tokens": cost,
        "thinking_fraction": float((1 - weights).mean()),
        "token_saving_pct": 100 * (1 - cost / t_cost),
        "accuracy_drop_pp": 100 * (t_acc - a),
        "gain_over_random_pp": gain_over_random(a, cost, nt_acc, t_acc, nt_cost, t_cost),
    }


def bootstrap_indices(n, samples=2000, seed=2027):
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def bootstrap_auroc(labels, scores, indices):
    y = np.asarray(labels)[indices]
    ranks = rankdata(np.asarray(scores)[indices], method="average", axis=1)
    positives = y.sum(1)
    negatives = y.shape[1] - positives
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((ranks * y).sum(1) - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap_metrics(data, weights, indices, scores=None):
    correct, tokens = outcomes(data, weights)
    a = correct[indices].mean(1)
    cost = tokens[indices].mean(1)
    nt_acc, t_acc = data["a_nt"][indices].mean(1), data["a_t"][indices].mean(1)
    nt_cost, t_cost = data["g_nt"][indices].mean(1), data["g_t"][indices].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (cost - nt_cost) / (t_cost - nt_cost)
        gain = 100 * (a - ((1 - q) * nt_acc + q * t_acc))
        gain[(q < -1e-10) | (q > 1 + 1e-10) | (abs(t_cost - nt_cost) < 1e-12)] = np.nan
    result = {
        "accuracy": a,
        "mean_tokens": cost,
        "token_saving_pct": 100 * (1 - cost / t_cost),
        "accuracy_drop_pp": 100 * (t_acc - a),
        "gain_over_random_pp": gain,
        "thinking_fraction": (1 - weights)[indices].mean(1),
    }
    if scores is not None:
        result["auroc"] = bootstrap_auroc(data["a_nt"], scores, indices)
    return result


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(finite, [2.5, 97.5]))


def with_intervals(point, bootstrap):
    result = dict(point)
    for name, values in bootstrap.items():
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = interval(values)
        result[f"{name}_bootstrap_valid"] = int(np.isfinite(values).sum())
    return result
