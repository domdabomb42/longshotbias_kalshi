"""Bias-correction probability models."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class BiasModel:
    base: IsotonicRegression
    by_category: Dict[str, IsotonicRegression]
    by_structure: Dict[str, IsotonicRegression]
    category_counts: Dict[str, int]
    structure_counts: Dict[str, int]
    blend_k: float = 200.0
    logistic: LogisticRegression | None = None

    def predict_one(self, implied_prob: float, category: str | None, structure: str | None) -> float:
        q_base = float(self.base.predict([implied_prob])[0])
        q_cat = None
        q_struct = None
        w_cat = 0.0
        w_struct = 0.0

        if category and category in self.by_category:
            q_cat = float(self.by_category[category].predict([implied_prob])[0])
            n_cat = float(self.category_counts.get(category, 0))
            w_cat = n_cat / (n_cat + self.blend_k) if n_cat > 0 else 0.0

        if structure and structure in self.by_structure:
            q_struct = float(self.by_structure[structure].predict([implied_prob])[0])
            n_struct = float(self.structure_counts.get(structure, 0))
            w_struct = n_struct / (n_struct + self.blend_k) if n_struct > 0 else 0.0

        total = w_cat + w_struct
        if total >= 1.0:
            if total > 0:
                w_cat /= total
                w_struct /= total
            w_base = 0.0
        else:
            w_base = 1.0 - total

        q = w_base * q_base
        if q_cat is not None:
            q += w_cat * q_cat
        if q_struct is not None:
            q += w_struct * q_struct
        return float(np.clip(q, 0.0, 1.0))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = []
        for _, row in df.iterrows():
            preds.append(self.predict_one(row["implied_prob"], row.get("category_mapped"), row.get("structure")))
        return pd.Series(preds, index=df.index, name="q_hat")


def train_isotonic(df: pd.DataFrame) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(df["implied_prob"], df["outcome"])
    return model


def train_models(
    observations: pd.DataFrame,
    min_samples: int = 200,
) -> BiasModel:
    df = observations.dropna(subset=["implied_prob", "outcome"]).copy()
    base = train_isotonic(df)

    by_category: Dict[str, IsotonicRegression] = {}
    category_counts: Dict[str, int] = {}
    for category, group in df.groupby("category_mapped"):
        if len(group) >= min_samples:
            by_category[category] = train_isotonic(group)
            category_counts[category] = int(len(group))

    by_structure: Dict[str, IsotonicRegression] = {}
    structure_counts: Dict[str, int] = {}
    for structure, group in df.groupby("structure"):
        if len(group) >= min_samples:
            by_structure[structure] = train_isotonic(group)
            structure_counts[structure] = int(len(group))

    logistic = None
    if df["outcome"].nunique() >= 2:
        logistic = LogisticRegression()
        logistic.fit(df[["implied_prob"]], df["outcome"])

    return BiasModel(
        base=base,
        by_category=by_category,
        by_structure=by_structure,
        category_counts=category_counts,
        structure_counts=structure_counts,
        logistic=logistic,
    )


def walk_forward_validation(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    df = df.dropna(subset=["implied_prob", "outcome", "close_ts"]).copy()
    df = df.sort_values("close_ts")
    if len(df) < n_splits + 1:
        return pd.DataFrame(
            [
                {
                    "split": 1,
                    "brier": np.mean((df["outcome"] - df["implied_prob"]) ** 2) if len(df) else np.nan,
                    "mae": np.mean(np.abs(df["outcome"] - df["implied_prob"])) if len(df) else np.nan,
                    "n_test": len(df),
                }
            ]
        )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for split, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        model = train_isotonic(train_df)
        preds = model.predict(test_df["implied_prob"])
        brier = np.mean((test_df["outcome"].values - preds) ** 2)
        mae = np.mean(np.abs(test_df["outcome"].values - preds))
        results.append({"split": split, "brier": brier, "mae": mae, "n_test": len(test_df)})
    return pd.DataFrame(results)


def build_correction_curves(model: BiasModel, categories: Iterable[str], structures: Iterable[str]) -> pd.DataFrame:
    grid = np.linspace(0.01, 0.99, 99)
    records: List[Dict[str, float | str]] = []
    for p in grid:
        records.append({"segment": "base", "implied_prob": p, "q_hat": model.base.predict([p])[0]})
    for category in categories:
        if category in model.by_category:
            for p in grid:
                records.append(
                    {"segment": f"category:{category}", "implied_prob": p, "q_hat": model.by_category[category].predict([p])[0]}
                )
    for structure in structures:
        if structure in model.by_structure:
            for p in grid:
                records.append(
                    {"segment": f"structure:{structure}", "implied_prob": p, "q_hat": model.by_structure[structure].predict([p])[0]}
                )
    return pd.DataFrame(records)


def save_model(model: BiasModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> BiasModel:
    return joblib.load(path)
