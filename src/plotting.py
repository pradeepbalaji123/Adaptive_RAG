"""Publication-oriented experiment plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def create_research_plots(
    summary: Sequence[dict[str, Any]],
    candidate_results: Sequence[dict[str, Any]],
    ablations: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    paths: list[Path] = []

    summary_frame = pd.DataFrame(summary)
    quality = summary_frame.melt(
        id_vars="system", value_vars=["exact_match", "f1", "retrieval_hit"],
        var_name="metric", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(data=quality, x="system", y="score", hue="metric", ax=ax)
    ax.set(title="Fixed vs Adaptive vs Oracle", xlabel="", ylabel="Score", ylim=(0, 1))
    fig.tight_layout()
    path = output / "system_quality.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    candidates = pd.DataFrame(candidate_results)
    selected = candidates[candidates["selected"].astype(bool)]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    order = list(selected["candidate_id"].value_counts().index)
    sns.countplot(data=selected, x="candidate_id", order=order, ax=ax, color="#4C78A8")
    ax.set(title="Adaptive Configuration Selection", xlabel="Candidate", ylabel="Questions")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = output / "selected_configuration_distribution.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    signal_frame = candidates[[
        "retrieval_confidence_normalized", "semantic_confidence_normalized",
        "context_consistency_normalized", "combined_score",
    ]].rename(columns={
        "retrieval_confidence_normalized": "R'", "semantic_confidence_normalized": "(1-U)'",
        "context_consistency_normalized": "C'", "combined_score": "J",
    }).melt(var_name="signal", value_name="score")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.violinplot(data=signal_frame, x="signal", y="score", inner="quartile", cut=0, ax=ax)
    ax.set(title="Reference-Free Signal Distributions", xlabel="", ylabel="Normalized score", ylim=(0, 1))
    fig.tight_layout()
    path = output / "signal_distributions.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    ablation_frame = pd.DataFrame(ablations).melt(
        id_vars="ablation", value_vars=["exact_match", "f1", "retrieval_hit"],
        var_name="metric", value_name="score",
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=ablation_frame, x="ablation", y="score", hue="metric", ax=ax)
    ax.set(title="Signal Ablation Comparison", xlabel="", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = output / "ablation_comparison.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    correlation_frame = pd.DataFrame(correlations).pivot(index="signal", columns="target", values="spearman")
    annotation = lambda value: "NA" if pd.isna(value) else f"{value:.2f}"
    annotations = (
        correlation_frame.map(annotation)
        if hasattr(correlation_frame, "map")
        else correlation_frame.applymap(annotation)
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    sns.heatmap(
        correlation_frame.fillna(0.0).astype(float), annot=annotations, fmt="", center=0,
        vmin=-1, vmax=1, cmap="vlag", ax=ax,
    )
    ax.set(title="Spearman Correlation with Answer Quality", xlabel="Target", ylabel="Signal")
    fig.tight_layout()
    path = output / "correlation_heatmap.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    latency = summary_frame.melt(
        id_vars="system",
        value_vars=["average_retrieval_latency", "average_generation_latency", "average_total_latency"],
        var_name="metric", value_name="seconds",
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.barplot(data=latency, x="system", y="seconds", hue="metric", ax=ax)
    ax.set(title="Latency Comparison", xlabel="", ylabel="Seconds")
    fig.tight_layout()
    path = output / "latency_comparison.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    return paths
