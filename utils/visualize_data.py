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
