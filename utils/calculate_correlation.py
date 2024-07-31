#!/usr/bin/env python

from scipy.stats import pearsonr
import pandas as pd

def calculate_correlations(X: pd.DataFrame, y: pd.Series):
    """
    Calculate the correlations between features and the target variable.
    
    :param X: DataFrame of features
    :param y: Series of the target variable
    :return: Dictionary of features and their correlations with the target
    """
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

def get_top_correlated_features(correlations: dict, top_n: int = 10):
    """
    Get the top N features with the highest correlation with the target.
    
    :param correlations: Dictionary of feature correlations
    :param top_n: Number of top features to select
    :return: List of top correlated features
    """
    sorted_correlations = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    return sorted_correlations[:top_n]
