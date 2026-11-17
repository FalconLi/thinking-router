"""Generate publication-exportable figures only from persisted CSV tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, parser, run_location

COLORS = {"hidden": "#1665ad", "tfidf": "#e07a19", "length": "#3a915c", "random": "#808080"}


def make_figures(config):
    root = run_location(config, "results")
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name, title):
        fig.suptitle(title)
        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(destination / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    endpoints_path = root / "tables/e1_endpoints.csv"
    if endpoints_path.exists():
        endpoints = pd.read_csv(endpoints_path)
        endpoints = endpoints[endpoints.method != "oracle"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, metric, label in zip(
            axes, ["accuracy", "mean_tokens"], ["Answer accuracy", "Mean generated tokens"], strict=True
        ):
            values = endpoints[metric].to_numpy()
            errors = np.array(
                [values - endpoints[f"{metric}_ci_low"], endpoints[f"{metric}_ci_high"] - values]
            )
            ax.bar(
                endpoints.method.str.replace("_", " "),
                values,
                yerr=np.maximum(errors, 0),
                color=["#1665ad", "#e07a19"],
            )
            ax.set_ylabel(label)
        save(fig, "e1_mode_tradeoff", "Two generation modes")
    curves_path = root / "curves/e2_routing_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        selected = pd.read_csv(root / "tables/e2_selected_policies.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in curves.groupby("method"):
            group = group.sort_values("mean_tokens")
            ax.plot(group.mean_tokens, group.accuracy, label=method, color=COLORS.get(method), alpha=0.75)
        for _, row in selected.iterrows():
            ax.errorbar(
                row.mean_tokens,
                row.accuracy,
                fmt="o",
                color=COLORS.get(row.method, "black"),
                xerr=[
                    [max(0, row.mean_tokens - row.mean_tokens_ci_low)],
                    [max(0, row.mean_tokens_ci_high - row.mean_tokens)],
                ],
                yerr=[
                    [max(0, row.accuracy - row.accuracy_ci_low)],
                    [max(0, row.accuracy_ci_high - row.accuracy)],
                ],
            )
            if row.method in ("oracle", "always_thinking", "never_thinking"):
                ax.annotate(
                    row.method.replace("_", " "),
                    (row.mean_tokens, row.accuracy),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(xlabel="Mean generated tokens", ylabel="Answer accuracy")
        ax.legend()
        save(fig, "e2_accuracy_tokens", "Routing trade-off; marked policies selected on validation")
    learning_path = root / "tables/e3_learning_curve_summary.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric, label in zip(
            axes,
            ["auroc", "accuracy", "token_saving_pct"],
            ["Classifier AUROC", "Answer accuracy", "Token saving (%)"],
            strict=True,
        ):
            for method, group in learning.groupby("method"):
                group = group.sort_values("training_size")
                mean = group[f"{metric}_mean"].to_numpy()
                low, high = [], []
                for _, row in group.iterrows():
                    if row.uncertainty_type == "test_bootstrap":
                        low.append(max(0, row[f"{metric}_mean"] - row.get(f"{metric}_ci_low", np.nan)))
                        high.append(max(0, row.get(f"{metric}_ci_high", np.nan) - row[f"{metric}_mean"]))
                    else:
                        low.append(row[f"{metric}_std"])
                        high.append(row[f"{metric}_std"])
                ax.errorbar(
                    group.training_size,
                    mean,
                    yerr=[low, high],
                    marker="o",
                    label=method,
                    color=COLORS[method],
                )
            ax.set(xscale="log", xlabel="Training questions (+ fixed validation set)", ylabel=label)
            ax.set_xticks(
                sorted(learning.training_size.unique()), labels=sorted(learning.training_size.unique())
            )
        selected_path = root / "tables/e2_selected_policies.csv"
        if selected_path.exists():
            selected = pd.read_csv(selected_path).set_index("method")
            axes[1].axhline(selected.loc["always_thinking", "accuracy"], color="grey", linestyle="--")
            if "auroc" in selected:
                axes[0].axhline(selected.loc["length", "auroc"], color=COLORS["length"], linestyle="--")
        axes[0].legend()
        save(fig, "e3_data_efficiency", "Error bars: subset SD; full pool: test bootstrap 95% CI")
    latency_path = root / "tables/e4_latency_summary.csv"
    if latency_path.exists():
        frame = pd.read_csv(latency_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        routing = frame.mean_routing_ms / 1000
        ax.bar(frame.method, routing, label="Routing")
        ax.bar(frame.method, frame.mean_generation_seconds, bottom=routing, label="Generation")
        ax.set_ylabel("Mean end-to-end seconds")
        ax.legend()
        save(fig, "e4_runtime", "Live execution, including routing overhead")
    print({"figures": str(destination)})


def main():
    args = parser(__doc__).parse_args()
    make_figures(load_config(args.config))


if __name__ == "__main__":
    main()
