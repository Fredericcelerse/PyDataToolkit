#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_data(data: pd.DataFrame):
    """
    Normalize the data using StandardScaler.
    
    :param data: DataFrame to normalize
    :return: Normalized DataFrame
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)
    
    return data_scaled_df
  
