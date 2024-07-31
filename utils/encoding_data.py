#!/usr/bin/env python

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_columns(data: pd.DataFrame, categorical_cols: list, exclude_columns: list = ['SalePrice']):
    """
    Encode categorical columns using one-hot encoding.
    
    :param data: DataFrame to encode
    :param categorical_cols: List of categorical columns to encode
    :param exclude_columns: List of columns to exclude from encoding (default is ['SalePrice'])
    :return: Encoded DataFrame
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'  # Leave the other columns unchanged
    )
    
    data_encoded = column_transformer.fit_transform(data.drop(columns=exclude_columns))
    data_encoded_df = pd.DataFrame(data_encoded, columns=column_transformer.get_feature_names_out())
    
    # Optionally add back the excluded columns
    for col in exclude_columns:
        if col in data.columns:
            data_encoded_df[col] = data[col].values
    
    return data_encoded_df
