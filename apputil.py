import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean", default_category=None):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.default_category = default_category
        self.group_values = None
        self.default_value = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)

        df = X_df.copy()
        df["target"] = y_series

        # Compute grouped estimates
        if self.estimate == "mean":
            grouped = df.groupby(list(X_df.columns))["target"].mean()
        else:
            grouped = df.groupby(list(X_df.columns))["target"].median()

        self.group_values = grouped.to_dict()

    def predict(self, X_):
        X_df = pd.DataFrame(X_)
        predictions = []

        for _, row in X_df.iterrows():
            key = tuple(row)

            if len(row) == 1:
                key = row.iloc[0]

            value = self.group_values.get(key, np.nan)

            # If missing, use default fallback
            if pd.isna(value):
                value = self.default_value

            predictions.append(value)

        return np.array(predictions)