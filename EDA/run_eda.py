from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_training.data_pipeline import add_target_columns, load_panel_dataframe

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_summary_tables(panel_df: pd.DataFrame) -> None:
    numeric_cols = panel_df.select_dtypes("number").columns
    summary = panel_df[numeric_cols].describe().transpose()
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    missing = panel_df[numeric_cols].isna().mean().sort_values(ascending=False)
    missing.to_csv(OUTPUT_DIR / "missing_ratio.csv")


def plot_target_distribution(panel_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(panel_df["target_value"], bins=50, kde=True, color="#1b5e20")
    plt.title("Distribution of Next-Quarter Net Sales")
    plt.xlabel("Net sales (shifted)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png")
    plt.close()


def plot_time_series(panel_df: pd.DataFrame) -> None:
    time_series = (
        panel_df.groupby("date")["net_sales"]
        .mean()
        .dropna()
        .reset_index()
    )
    plt.figure(figsize=(10, 4))
    plt.plot(time_series["date"], time_series["net_sales"], color="#0d47a1")
    plt.title("Average Net Sales Over Time")
    plt.xlabel("Quarter")
    plt.ylabel("Average net sales")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_net_sales_timeseries.png")
    plt.close()


def plot_feature_heatmap(panel_df: pd.DataFrame, top_k: int = 12) -> None:
    corr_series = (
        panel_df.corr(numeric_only=True)["target_value"]
        .dropna()
        .abs()
        .sort_values(ascending=False)
    )
    top_features = corr_series.head(top_k + 1).index  # include target_value
    corr_matrix = panel_df[top_features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        square=True,
    )
    plt.title("Top Feature Correlations with Target")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_heatmap.png")
    plt.close()


def main() -> None:
    panel_df = load_panel_dataframe()
    panel_df = add_target_columns(panel_df, target_col="net_sales")

    generate_summary_tables(panel_df)
    plot_target_distribution(panel_df)
    plot_time_series(panel_df)
    plot_feature_heatmap(panel_df)


if __name__ == "__main__":
    main()

