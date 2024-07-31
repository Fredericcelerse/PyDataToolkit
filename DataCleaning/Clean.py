#!/usr/bin/env python

import pandas as pd
from sklearn.impute import SimpleImputer

def check_missing_data(data: pd.DataFrame):
    """
    Check for missing data in train and test datasets.
    
    :param data: DataFrame
    :return: DataFrame of missing data summary
    """
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data.sum(axis=1) > 0]
    return missing_data

def clean_missing_data(missing_data: pd.DataFrame):
    """
    Clean the missing data by dropping duplicates.
    
    :param missing_data: DataFrame of missing data summary
    :return: Cleaned DataFrame of missing data summary
    """
    cleaned_missing_data = missing_data.drop_duplicates()
    return cleaned_missing_data

def drop_columns_with_missing_data(data: pd.DataFrame, threshold: float = 0.5):
    """
    Drop columns where missing data are higher than the defined threshold

    :param dat: DataFrame
    :param threshold: Threshold we define manually
    :return: Cleaned data
    """
    data = data[data.columns[data.isnull().mean() < threshold]]
    return data

def impute_missing_values(data: pd.DataFrame):
    """
    Impute missing values in the DataFrame.
    Float and In are filled using the mean approach
    Object are filled using the most frequent approach
    
    :param data: DataFrame to impute
    :return: DataFrame with imputed values
    """
    # Impute missing values for numerical columns
    numeric_cols = data.select_dtypes(include=[float, int]).columns
    imputer_mean = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer_mean.fit_transform(data[numeric_cols])
    
    # Impute missing values for categorical columns
    categorical_cols = data.select_dtypes(include=[object]).columns
    imputer_mode = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = imputer_mode.fit_transform(data[categorical_cols])
    
    return data

