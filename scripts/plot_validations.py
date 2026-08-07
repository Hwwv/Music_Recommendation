#!/usr/bin/env python3
"""Visualize the validation-time hyperparameter sweeps for every model variant.

Produces:
  - one standalone panel PNG per model
  - one compact combined 2x4 grid PNG with all standalong plots and a combined comparison in a single figure
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use the paths of the validation results for plotting
RESULT_FILES = {
    "cbm":                Path("artifacts/cbm/cbm_eval20_validation_v1.json"),
    "multi_cbm":          Path("artifacts/multicbm/multicbm_eval20_validation_v3.json"),
    "cf":                 Path("artifacts/cf/item-knn_validation_v1.json"),
    "als":                Path("artifacts/cf/als_validation_v1.json"),
    "hybrid":             Path("artifacts/hybrid/hybrid_validation_v1.json"),
    "cbm_genre":          Path("artifacts2/cbm/cbm_eval20_validation_v1.json"),
    "multi_cbm_genre":    Path("artifacts2/multicbm2/multicbm_eval20_validation_genre_v1.json"),
}

TITLES = {
    "cbm": "CBM (content-based)",
    "multi_cbm": "Multi-Interest CBM",
    "cf": "Item-KNN (CF)",
    "als": "Implicit ALS",
    "hybrid": "Hybrid (CF + CBM)",
    "cbm_genre": r"CBM genre $\bf{(V2\ Genre\ Dataset)}$",
    "multi_cbm_genre": r"Multi-CBM genre $\bf{(V2\ Genre\ Dataset)}$",
}

# hyperparameters to name in each panel's "best config" annotation
HYPERPARAM_KEYS = {
    "cbm": ("alpha",),
    "multi_cbm": ("alpha", "gw", "km"),
    "cf": ("alpha", "neighbours", "min_cooccurrence", "weighting"),
    "als": ("alpha", "factors", "iterations", "regularization"),
    "cbm_genre": ("alpha",),
    "multi_cbm_genre": ("alpha", "gw", "km"),
}

K_PRIMARY = 20  # metric suffix used for selection / plotting (falls back to @10)
N_V2_MODEL = 2
OUTPUT_DIR = Path("artifacts/figures")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex": False,
    "mathtext.default": "regular",
})


def _metric(metrics: dict, name: str, k: int = K_PRIMARY) -> float | None:
    for candidate_k in (k, 20, 10):
        key = f"{name}@{candidate_k}"
        if key in metrics:
            return metrics[key]
    return None


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _parse_config_key(key: str) -> dict:
    """'0.95' -> {'alpha': 0.95} for CBM model; 'alpha_0.90_gw_0.30_km_16' -> {'alpha':0.9,'gw':0.3,'km':16} for Multi-CBM model."""
    if _looks_numeric(key):
        return {"alpha": float(key)}
    parts = key.split("_")
    parsed: dict[str, float] = {}
    i = 0
    while i < len(parts) - 1:
        name, value = parts[i], parts[i + 1]
        if _looks_numeric(value):
            parsed[name] = float(value)
        i += 2
    return parsed


def load_results(path: Path) -> list[dict]:
    """Flatten either JSON shape into a list of {**hyperparams, recall, ndcg, hit_rate, coverage}."""
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    records: list[dict] = []
    if isinstance(raw, dict) and "runs" in raw:
        for run in raw["runs"]:
            metrics = run["metrics"]
            config = dict(run.get("configuration", {}))
            if "cf_weight" in run:
                config["cf_weight"] = run["cf_weight"]
            records.append({
                **config,
                "recall": _metric(metrics, "recall"),
                "ndcg": _metric(metrics, "ndcg"),
                "hit_rate": _metric(metrics, "hit_rate"),
                "coverage": _metric(metrics, "catalog_coverage"),
            })
    elif isinstance(raw, dict):
        for key, metrics in raw.items():
            config = _parse_config_key(key)
            records.append({
                **config,
                "recall": _metric(metrics, "recall"),
                "ndcg": _metric(metrics, "ndcg"),
                "hit_rate": _metric(metrics, "hit_rate"),
                "coverage": _metric(metrics, "catalog_coverage"),
            })
    else:
        raise ValueError(f"unrecognized results schema in {path}")

    for record in records:
        if record["recall"] is None:
            raise ValueError(f"could not find a recall@k metric in {path}")
    return records


def config_parts(record: dict, keys_to_show: tuple[str, ...]) -> list[str]:
    parts = []
    for key in keys_to_show:
        value = record.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return parts or ["config"]


# Plotting functions
def plot_sweep_panel(ax, records: list[dict], title: str, hyperparam_keys: tuple[str, ...]) -> None:
    """Recall@k across every validation configuration, sorted, best one highlighted
    and annotated with its other metrics."""
    records_sorted = sorted(records, key=lambda r: r["recall"])
    recalls = [r["recall"] for r in records_sorted]
    x = np.arange(len(records_sorted))
    best_idx = int(np.argmax(recalls))

    colors = ["#a9c6e8"] * len(records_sorted)
    colors[best_idx] = "#d1495b"

    # Auto-zoom the y-axis when recall values are tightly clustered
    value_min, value_max = min(recalls), max(recalls)
    value_range = value_max - value_min
    mean_recall = float(np.mean(recalls))
    tightly_clustered = value_range < 0.15 * mean_recall if mean_recall else False
    if tightly_clustered:
        pad = max(value_range, 1e-9) * 0.25
        y_bottom = max(0.0, value_min - pad)
        y_top = value_max + pad * 3.5  # extra headroom for the annotation box
        ax.set_ylim(y_bottom, y_top)
        base = y_bottom
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
    else:
        base = 0.0

    ax.bar(x, [r - base for r in recalls], bottom=base, color=colors, width=0.85, edgecolor="none")
    ax.axhline(mean_recall, color="#444444", linestyle="--", linewidth=0.8)

    best = records_sorted[best_idx]
    lines = [f"mean={mean_recall:.4f}", "best:"] + [f"  {p}" for p in config_parts(best, hyperparam_keys)]
    for label, key in (("ndcg", "ndcg"), ("hit_rate", "hit_rate"), ("coverage", "coverage")):
        if best.get(key) is not None:
            lines.append(f"{label}={best[key]:.4f}")
    box_y = 0.96 if tightly_clustered else 0.04
    box_va = "top" if tightly_clustered else "bottom"
    ax.annotate(
        "\n".join(lines), xy=(best_idx, recalls[best_idx]),
        xytext=(0.98, box_y), textcoords="axes fraction", fontsize=5, ha="right", va=box_va,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d1495b", lw=0.6, alpha=0.9),
    )

    if "genre" in title:
        ax.set_facecolor('#f0f0f0')

    ax.set_title(title)
    ax.set_ylabel(f"recall@{K_PRIMARY}")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(records_sorted)} configs (sorted)")
    ax.spines[["top", "right"]].set_visible(False)


def plot_hybrid_trend_panel(ax, records: list[dict], title: str, chosen_weight:float = 0.5) -> None:
    """recall@k and catalog_coverage@k vs cf_weight, dual y-axis."""
    records_sorted = sorted(records, key=lambda r: r["cf_weight"])
    cf_weights = [r["cf_weight"] for r in records_sorted]
    recalls = [r["recall"] for r in records_sorted]
    coverages = [r["coverage"] for r in records_sorted]

    color_recall, color_coverage = "#d1495b", "#2a6f97"

    ax.plot(cf_weights, recalls, marker="o", ms=3, lw=1.3, color=color_recall,
            label=f"recall@{K_PRIMARY}")
    ax.set_xlabel("cf_weight")
    ax.set_ylabel(f"Metric Values")
    ax.tick_params(axis="y")
    ax.invert_xaxis()

    ax.plot(cf_weights, coverages, marker="s", ms=3, lw=1.3, color=color_coverage,
             linestyle="--", label=f"catalog_coverage@{K_PRIMARY}")

    try:
        chosen_index = cf_weights.index(chosen_weight)
    except ValueError:
        chosen_index = min(range(len(cf_weights)), key=lambda i: abs(cf_weights[i]-chosen_weight))

    ax.scatter([chosen_weight], recalls[chosen_index], color=color_recall, s=35,
               zorder=5, edgecolor="black", linewidth=0.5)
    ax.annotate(
        f"chosen cf_weight={chosen_weight:g}\nrecall={recalls[chosen_index]:.4f}\ncoverage={coverages[chosen_index]:.4f}",
        xy=(cf_weights[chosen_index], recalls[chosen_index]), xytext=(0.7, 0.7), textcoords="axes fraction",
        fontsize=6, ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_recall, lw=0.6, alpha=0.9,),
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1, bbox_to_anchor=(-0.02, 0.05), loc="lower left", frameon=False)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)


def plot_summary_panel(ax, best_by_model: dict[str, dict]) -> None:
    """Bonus panel: best validation recall@k achieved by each model, side by side."""
    names = list(best_by_model.keys())[:-N_V2_MODEL]
    values = [best_by_model[name]["recall"] for name in names]
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    bars = ax.barh([TITLES.get(n, n) for n in names], values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.4f}", va="center", fontsize=6.5)
    ax.set_xlabel(f"best recall@{K_PRIMARY}")
    ax.set_title("Best validation recall by model")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = {name: load_results(path) for name, path in RESULT_FILES.items()}
    best_by_model = {name: max(recs, key=lambda r: r["recall"]) for name, recs in all_records.items()}

    for name, records in all_records.items():
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        if name == "hybrid":
            plot_hybrid_trend_panel(ax, records, TITLES[name])
        else:
            plot_sweep_panel(ax, records, TITLES[name], HYPERPARAM_KEYS[name])
        fig.tight_layout()
        out = OUTPUT_DIR / f"panel_{name}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {out}")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    plot_summary_panel(ax, best_by_model)
    fig.tight_layout()
    out = OUTPUT_DIR / "panel_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

    # 2*4 combined panels
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    axes = axes.ravel()
    order = ["cbm", "multi_cbm", "cf", "als", "hybrid", "cbm_genre", "multi_cbm_genre"]
    for ax, name in zip(axes, order):
        records = all_records[name]
        if name == "hybrid":
            plot_hybrid_trend_panel(ax, records, TITLES[name])
        else:
            plot_sweep_panel(ax, records, TITLES[name], HYPERPARAM_KEYS[name])
    plot_summary_panel(axes[7], best_by_model)

    fig.suptitle("Validation Results for Trainable Recommenders(CF: Sparse Item-KNN, ALS; CBM: CBM, Multi-CBM; hybrid)", fontweight='bold')
    fig.tight_layout()
    combined = OUTPUT_DIR / "validation_sweeps_combined.png"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {combined}")


if __name__ == "__main__":
    main()