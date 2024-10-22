# data_processing_tool.py

import pandas as pd
from typing import Dict
#include
def read_csv(file_path: str) -> pd.DataFrame:
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data by removing NaNs and outliers."""
    df = df.dropna()  # Remove missing values
    df = df[(df >= df.quantile(0.01)) & (df <= df.quantile(0.99))]  # Remove outliers
    return df

def transform_data(df: pd.DataFrame, transformation: str) -> pd.DataFrame:
    """Applies a transformation like normalization or standardization."""
    if transformation == 'normalize':
        return (df - df.min()) / (df.max() - df.min())
    elif transformation == 'standardize':
        return (df - df.mean()) / df.std()
    else:
        raise ValueError("Unknown transformation type")

def compute_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """Computes summary statistics like mean, median, and standard deviation."""
    return {
        'mean': df.mean().to_dict(),
        'median': df.median().to_dict(),
        'mode': df.mode().iloc[0].to_dict(),
        'variance': df.var().to_dict(),
        'std_dev': df.std().to_dict()
    }

# Example usage (you can remove this in production)
if __name__ == "__main__":
    df = read_csv('data.csv')
    df = clean_data(df)
    transformed_df = transform_data(df, 'normalize')
    stats = compute_statistics(transformed_df)
    print(stats)
