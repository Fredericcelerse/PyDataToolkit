#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlation_matrix(data: pd.DataFrame, method: str = 'pearson'):
    """
    Calculate the correlation matrix.
    
    :param data: DataFrame to calculate correlation matrix
    :param method: Correlation method (default is 'pearson')
    :return: Correlation matrix DataFrame
    """
    return data.corr(method=method)

def get_top_correlated_features(corr_matrix: pd.DataFrame, target: str, top_n: int = 10):
    """
    Get the top N features most correlated with the target variable.
    
    :param corr_matrix: Correlation matrix DataFrame
    :param target: Target variable name
    :param top_n: Number of top features to select
    :return: List of top correlated features
    """
    corr_with_target = corr_matrix[target].abs()
    top_features = corr_with_target.sort_values(ascending=False).head(top_n + 1).index.drop(target)  # Exclude target
    return top_features

def plot_correlation_heatmap(data: pd.DataFrame, features: list):
    """
    Plot a heatmap of the correlation matrix for the selected features.
    
    :param data: DataFrame containing the data
    :param features: List of features to include in the heatmap
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[features].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap (Top Features)')
    plt.show()
