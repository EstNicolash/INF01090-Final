import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Builds an isolated, non-destructive ColumnTransformer preprocessing pipe.
    Sychronized with the engineered features inside df_processed, ensuring
    proper scaling, categorical encoding, and pass-through flags logic.
    
    Returns:
        ColumnTransformer: Preprocessing configuration ready to transform features.
    """
    # 1. Continuous and ordinal indicators that require standardization
    numerical_features = [
        'latitude', 'longitude', 
        'shore_distance_log', 
        'feat_season_sine', 'feat_season_cosine'
    ]
    
    # 2. High and low cardinality nominal labels that require one-hot vectorization
    categorical_features = [
        'vessel_type', 'vessel_status', 'nearest_country', 
        'lat_bin', 'lon_bin'
    ]
    
    # 3. Micro-engineered binary triggers already encoded as 0 or 1 in int8.
    # We explicitly pass them through untransformed to preserve spatial/text semantics.
    passthrough_features = [
        'is_international_waters',
        'feat_is_near_port',
        'feat_time_recorded',
    ]
    
    # Atomic transformations inside structural sub-pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Fuse transformations into a unified memory-aligned pipeline configuration
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', passthrough_features)  
        ],
        remainder='drop' # Drops organizational metrics like 'year' which shouldn't enter training arrays
    )
    
    return preprocessor