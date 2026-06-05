from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
import numpy as np

def build_preprocessing_pipeline() -> ColumnTransformer:

    numerical_features = [
        'shore_distance_log',
        #'feat_season_sine', 'feat_season_cosine',
        'year'
    ]

    # Apenas colunas de baixa cardinalidade aqui
    low_cardinality_cat = [
        'vessel_type', 'vessel_status',
        'lat_bin', 'lon_bin'
    ]

    # nearest_country tratado separadamente com frequency encoding
    high_cardinality_cat = ['nearest_country', 'region_cell']

    passthrough_features = [
        'is_international_waters',
        'feat_is_near_port',
        'feat_time_recorded',
        'feat_vessel_moving',
    ]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    low_card_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Frequency encoding manual via FunctionTransformer
    # Conta a frequência de cada categoria e substitui pelo valor normalizado
    from sklearn.preprocessing import FunctionTransformer

    def frequency_encode(X):
        """Substitui cada categoria pela sua frequência relativa na coluna."""
        result = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            col = X[:, i].astype(str)
            vals, counts = np.unique(col, return_counts=True)
            freq_map = dict(zip(vals, counts / counts.sum()))
            result[:, i] = np.array([freq_map.get(v, 0.0) for v in col])
        return result

    high_card_transformer = Pipeline(steps=[
        ('freq_enc', FrequencyEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num',      numeric_transformer,   numerical_features),
            ('cat_low',  low_card_transformer,  low_cardinality_cat),
            ('cat_high', high_card_transformer, high_cardinality_cat),
            ('pass',     'passthrough',         passthrough_features),
        ],
        remainder='drop'
    )

    return preprocessor

from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):     
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        
        self.freq_maps_ = []
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i].astype(str)
            vals, counts = np.unique(col, return_counts=True)
            self.freq_maps_.append(
                dict(zip(vals, counts / counts.sum()))
            )
        return self

    def transform(self, X):
        X_arr = X.values if hasattr(X, 'values') else np.array(X)
        
        result = np.zeros((X_arr.shape[0], X_arr.shape[1]))
        for i, freq_map in enumerate(self.freq_maps_):
            col = X_arr[:, i].astype(str)
            result[:, i] = np.array([freq_map.get(v, 0.0) for v in col])
        return result