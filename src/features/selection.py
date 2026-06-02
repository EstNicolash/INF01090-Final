import pandas as pd
import numpy as np
import pandas.api.types as pdt
from sklearn.metrics import normalized_mutual_info_score

def compute_numerical_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a standard Pearson correlation matrix strictly for the 
    numerical continuous features within the piracy dataset.
    """
    # Filter strictly numerical features available in the current layout
    num_cols = [col for col in df.columns if pdt.is_numeric_dtype(df[col].dtype) and col != 'year']
    
    if not num_cols:
        return pd.DataFrame()
        
    # Standard linear correlation matrix computation
    corr_matrix = df[num_cols].corr(method='pearson')
    return corr_matrix

def compute_categorical_nmi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a pairwise Normalized Mutual Information (NMI) matrix 
    for categorical features, including correlation bounds against the target.
    """
    # Isolate categorical and structural string components
    cat_cols = [
        col for col in df.columns 
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == object or col == 'is_international_waters'
    ]
    
    # Exclude high-cardinality open text blocks or unique identifiers to prevent NMI inflation
    exclusions = ['date', 'time', 'attack_description', 'vessel_name', 'location_description']
    cat_cols = [c for c in cat_cols if c not in exclusions]
    
    n_features = len(cat_cols)
    nmi_matrix = np.zeros((n_features, n_features))
    
    # Compute pairwise NMI scores sequentially
    for i in range(n_features):
        for j in range(n_features):
            # Drop missing links on the fly for accurate joint distribution indexing
            valid_idx = df[cat_cols[i]].notna() & df[cat_cols[j]].notna()
            
            if valid_idx.any():
                nmi_matrix[i, j] = normalized_mutual_info_score(
                    df.loc[valid_idx, cat_cols[i]].astype(str),
                    df.loc[valid_idx, cat_cols[j]].astype(str),
                    average_method='arithmetic'
                )
            else:
                nmi_matrix[i, j] = 0.0
                
    df_nmi = pd.DataFrame(nmi_matrix, index=cat_cols, columns=cat_cols)
    return df_nmi


import pandas as pd
import numpy as np
import pandas.api.types as pdt
from sklearn.metrics import normalized_mutual_info_score

def compute_target_association(df: pd.DataFrame, target_col: str = 'attack_type') -> pd.DataFrame:
    """
    Computes the Normalized Mutual Information (NMI) of all engineered and
    categorical features strictly against the multi-class target variable.
    Returns a sorted DataFrame showcasing direct predictive strength.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    df_clean = df.dropna(subset=[target_col]).copy()
    target_series = df_clean[target_col].astype(str)
    
    # Identify all potential features, excluding raw text, administrative IDs or date logs
    exclusions = ['date', 'time', 'attack_description', 'vessel_name', 'location_description', target_col]
    feature_cols = [c for c in df_clean.columns if c not in exclusions]
    
    associations = []
    
    for col in feature_cols:
        # For continuous numerical features, we temporarily bin them using quantiles 
        # to calculate a stable semantic NMI without continuous density noise
        if pdt.is_numeric_dtype(df_clean[col].dtype) and col != 'year':
            try:
                # Discretize into 10 bins based on sample distribution
                binned_feature = pd.qcut(df_clean[col], q=10, labels=False, duplicates='drop').astype(str)
            except Exception:
                binned_feature = df_clean[col].astype(str)
        else:
            binned_feature = df_clean[col].astype(str)
            
        # Compute NMI against target
        nmi_score = normalized_mutual_info_score(binned_feature, target_series, average_method='arithmetic')
        
        associations.append({
            "Feature": col,
            "NMI_with_Target": nmi_score
        })
        
    df_res = pd.DataFrame(associations)
    return df_res.sort_values(by="NMI_with_Target", ascending=False).reset_index(drop=True)