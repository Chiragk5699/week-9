import pandas as pd
import numpy as np

class GroupEstimate:
    """
    A simple estimator that predicts target values based on group-level statistics 
    (mean or median) of the input features.

    Parameters
    ----------
    estimate : str, default="mean"
        Specifies which group statistic to use for prediction.
        Options:
            - "mean": use the mean of the target values in each group.
            - "median": use the median of the target values in each group.
    
    default_category : optional
        If provided, unseen combinations of features during prediction will 
        fallback to the overall mean or median of the target values.
        If None, unseen combinations return NaN.
    
    Attributes
    ----------
    group_values : dict
        Dictionary mapping tuples of feature values to their corresponding 
        group estimate (mean or median).

    default_value : float
        The overall mean or median of the target values, used as fallback 
        if `default_category` is set.
    """

    def __init__(self, estimate="mean", default_category=None):
        """
        Initialize the GroupEstimate object.
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be 'mean' or 'median'")
        
        self.estimate = estimate
        self.default_category = default_category
        self.group_values = None
        self.default_value = None

    def fit(self, X, y):
        """
        Fit the estimator by computing group-level statistics for each unique 
        combination of feature values.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature data used for grouping. Each row represents an observation, 
            and each column a feature.

        y : array-like
            Target values corresponding to each row in X.

        Returns
        -------
        self : GroupEstimate
            Returns the fitted estimator.
        """
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)

        df = X_df.copy()
        df["target"] = y_series

        # Compute grouped values
        if self.estimate == "mean":
            grouped = df.groupby(list(X_df.columns))["target"].mean()
            self.default_value = y_series.mean()
        else:
            grouped = df.groupby(list(X_df.columns))["target"].median()
            self.default_value = y_series.median()

        # ALWAYS store keys as tuples
        self.group_values = grouped.to_dict()

    def predict(self, X_):
        """
        Predict target values for new data based on group-level statistics.

        Parameters
        ----------
        X_ : array-like or DataFrame
            Feature data for which to generate predictions. Each row represents
            an observation, and each column a feature.

        Returns
        -------
        predictions : ndarray
            Array of predicted values. If a combination of feature values was 
            not seen during training:
                - Returns NaN if `default_category` is None.
                - Returns the overall mean or median if `default_category` is set.
        """
        X_df = pd.DataFrame(X_)
        predictions = []

        for _, row in X_df.iterrows():
            # ALWAYS use tuple key
            key = tuple(row.values)

            value = self.group_values.get(key, np.nan)

            # Only fallback if explicitly enabled
            if pd.isna(value) and self.default_category is not None:
                value = self.default_value

            predictions.append(value)

        return np.array(predictions)