#!/usr/bin/env python

import pandas as pd

def load_data_csv(path: str):
    """
    Load CSV files.
    
    :param path: Path to the CSV file
    :return: Tuple of panda DataFrame (data)
    """
    data = pd.read_csv(path)
    return data

def print_head(data: pd.DataFrame, n: int = 5):
    """
    Print the first n rows of the DataFrame.
    
    :param data: DataFrame to print
    :param n: Number of rows to print, default is 5
    """
    print(data.head(n))
