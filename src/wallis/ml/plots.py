from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_model_comparison(df: pd.DataFrame, metric: str, title: str, out_path: str | Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = df.sort_values(metric, ascending=False).copy()
    sns.barplot(data=plot_df, x="model", y=metric, hue="feature_set", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="feature set", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_top_features(df: pd.DataFrame, title: str, out_path: str | Path, top_n: int = 12) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("permutation importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
