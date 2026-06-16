from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(
    pipeline,
    X: pd.DataFrame,
    y,
    feature_names: list[str],
    random_state: int = 42,
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table.reset_index(drop=True)
