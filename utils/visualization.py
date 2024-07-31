#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_missing_data(data: pd.DataFrame):
    """
    Visualize missing data in the DataFrame using a heatmap.
    
    :param data: DataFrame to visualize
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap of missing data')
    plt.show()

def visualize_specific_data(data: pd.DataFrame, feature_name: str, bins: int = 20):
    """
    Visualize only one feature of a DataFrame.

    :param data: DataFrame to visualize
    :param feature_name: Name of the feature to visualize
    :param bins: Number of bins
    """
    sns.histplot(data[feature_name], bins=bins, kde=True)
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.show()

def visualize_multiple_data(data: pd.DataFrame, num_rows: int = None):
    """
    Visualize mutliple data at once. Optionally, only a subset of the data can be visualized.

    :param data: DataFrame to visualize
    :param num_rows: Number of rows to visualize. If None, visualize all data.
    """
    if num_rows is not None:
        data = data.head(num_rows)
    
    data.hist(figsize=(20, 15), bins=20)
    plt.tight_layout()
    plt.show()
    
