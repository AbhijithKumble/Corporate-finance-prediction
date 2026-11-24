from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [
    PROJECT_ROOT / "data" / "data_cleaned",
    PROJECT_ROOT / "data" / "other_data",
]


@dataclass
class PCAArtifacts:
    scaler: StandardScaler
    pca: PCA
    component_names: List[str]


def _sanitize_feature_name(raw_name: str) -> str:
    cleaned = raw_name.strip().lower()
    cleaned = cleaned.replace("&", "and")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def _load_feature_table(path: Path, existing: set[str]) -> pd.DataFrame:
    feature_name = _sanitize_feature_name(path.stem)
    if feature_name in existing:
        feature_name = f"{feature_name}_{path.parent.name}"
    existing.add(feature_name)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "company"})
    long_df = df.melt(
        id_vars="company",
        var_name="date",
        value_name=feature_name,
    )
    long_df["date"] = pd.to_datetime(long_df["date"].str.strip(), format="%b %Y", errors="coerce")
    long_df = long_df.dropna(subset=["date"]).copy()
    long_df["company"] = long_df["company"].astype(str).str.strip()
    return long_df


def load_panel_dataframe(min_presence: float = 0.3) -> pd.DataFrame:
    feature_frames: List[pd.DataFrame] = []
    seen: set[str] = set()
    for directory in DATA_DIRS:
        for file_path in sorted(directory.glob("*.csv")):
            feature_frames.append(_load_feature_table(file_path, seen))

    panel_df = reduce(
        lambda left, right: pd.merge(left, right, on=["company", "date"], how="outer"),
        feature_frames,
    )
    panel_df = panel_df.sort_values(["company", "date"]).reset_index(drop=True)

    data_cols = [col for col in panel_df.columns if col not in {"company", "date"}]
    presence = panel_df[data_cols].notna().mean()
    keep_cols = presence[presence >= min_presence].index.tolist()
    panel_df = panel_df[["company", "date", *keep_cols]]

    numeric_cols = keep_cols
    panel_df[numeric_cols] = (
        panel_df.groupby("company")[numeric_cols]
        .transform(lambda group: group.ffill().bfill())
    )
    panel_df[numeric_cols] = panel_df[numeric_cols].fillna(panel_df[numeric_cols].median())
    panel_df = panel_df.dropna(subset=numeric_cols, how="all")
    return panel_df


def add_target_columns(panel_df: pd.DataFrame, target_col: str = "net_sales") -> pd.DataFrame:
    if target_col not in panel_df.columns:
        raise KeyError(f"{target_col} not found in panel dataframe columns.")
    panel_df = panel_df.sort_values(["company", "date"]).reset_index(drop=True)
    panel_df["target_value"] = (
        panel_df.groupby("company")[target_col].shift(-1)
    )
    panel_df["target_date"] = (
        panel_df.groupby("company")["date"].shift(-1)
    )
    panel_df = panel_df.dropna(subset=["target_value", "target_date"]).reset_index(drop=True)
    return panel_df


def add_pca_features(
    panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_end: str = "2022-01-01",
    variance_threshold: float = 0.9,
) -> Tuple[pd.DataFrame, PCAArtifacts]:
    feature_cols = [col for col in feature_cols if col not in {"company", "date", "target_value", "target_date"}]
    panel_df = panel_df.copy()
    train_mask = panel_df["date"] < pd.Timestamp(train_end)
    scaler = StandardScaler()
    scaled_all = scaler.fit_transform(panel_df[feature_cols])
    scaled_train = scaled_all[train_mask.to_numpy()]

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(scaled_train)
    transformed = pca.transform(scaled_all)
    component_names = [f"pc_{idx+1}" for idx in range(transformed.shape[1])]
    for idx, name in enumerate(component_names):
        panel_df[name] = transformed[:, idx]

    artifacts = PCAArtifacts(scaler=scaler, pca=pca, component_names=component_names)
    return panel_df, artifacts


def build_sequences(
    panel_df: pd.DataFrame,
    feature_cols: Sequence[str],
    sequence_length: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    target_dates: List[pd.Timestamp] = []

    for _, group in panel_df.groupby("company"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=float)
        target_vals = group["target_value"].to_numpy(dtype=float)
        target_periods = group["target_date"].to_list()

        for idx in range(sequence_length - 1, len(group)):
            target_val = target_vals[idx]
            target_dt = target_periods[idx]
            if np.isnan(target_val) or pd.isna(target_dt):
                continue
            seq = values[(idx - sequence_length + 1) : idx + 1]
            if np.isnan(seq).any():
                continue
            sequences.append(seq)
            targets.append(target_val)
            target_dates.append(target_dt)

    return np.array(sequences), np.array(targets), target_dates


def split_by_date(
    target_dates: Sequence[pd.Timestamp],
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.array([ts <= pd.Timestamp(train_end) for ts in target_dates])
    val_mask = np.array(
        [pd.Timestamp(train_end) < ts <= pd.Timestamp(val_end) for ts in target_dates]
    )
    test_mask = ~(train_mask | val_mask)
    return train_mask, val_mask, test_mask


