#!/usr/bin/env python

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
