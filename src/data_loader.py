import pandas as pd
import numpy as np

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw maritime piracy dataset from a CSV file.
    Applies baseline data type optimizations to comply with Data-Oriented Design,
    extracts temporal fields, and structures categorical variables.

    Args:
        file_path (str): The path to the raw piracy_attacks.csv file.

    Returns:
        pd.DataFrame: A memory-optimized Pandas DataFrame containing a 'year' column.
    """
    # Load dataset keeping raw structures intact first
    df = pd.read_csv(file_path)
    
    # Vectorized extraction of the 4-character year prefix from the string date
    # Then cast directly to low-precision integer to optimize memory blocks
    if 'date' in df.columns:
        df['year'] = df['date'].astype(str).str[:4].astype(np.int16)
    
    # Optimize standard object columns with low cardinality into categorical types
    categorical_cols = ['attack_type', 'vessel_type', 'vessel_status']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df
