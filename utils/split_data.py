#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets.
    
    :param X: Features DataFrame
    :param y: Target Series
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Seed used by the random number generator
    :return: Tuple containing training and testing sets (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
