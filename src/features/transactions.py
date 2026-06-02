"""features.transactions – pure, pipe-compatible data-transformation stubs.

Each function in this module is a *transaction*: it receives an immutable
DataFrame, applies a single, well-scoped transformation, and returns a **new**
DataFrame without mutating the input.  The functions are designed to be
composed via ``pd.DataFrame.pipe()``.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




def tx_spatial_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise latitude/longitude coordinates into spatial grid bins.

    Projects each incident location onto a configurable grid and adds
    ``lat_bin`` and ``lon_bin`` categorical columns that group incidents by
    geographic region.
    """
    df_out = df.copy()
    
    # Using a standard 5-degree grid resolution (approx 550km bins at the equator)
    # This creates discrete regional clusters that tree models can easily split on
    grid_size = 5.0
    
    if 'latitude' in df_out.columns and 'longitude' in df_out.columns:
        df_out['lat_bin'] = (df_out['latitude'] // grid_size).fillna(-999).astype(int).astype(str).astype('category')
        df_out['lon_bin'] = (df_out['longitude'] // grid_size).fillna(-999).astype(int).astype(str).astype('category')
        
    return df_out


def tx_nlp_vessel_mining(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction: Mines missing historical 'vessel_type' data using a shallow
    NLP pipeline (TF-IDF + Logistic Regression) trained on hybrid textual metadata
    fusing both 'vessel_name' and 'attack_description'.
    Implements a non-destructive functional update on the categorical block.
    """
    df_out = df.copy()
    
    # Clean and fill missing slots for both text fields to build a reliable corpus
    clean_names = df_out['vessel_name'].astype(str).fillna('').str.upper()
    clean_descs = df_out['attack_description'].astype(str).fillna('').str.lower()
    
    # Feature Fusion: Concatenate metadata tokens to amplify the classifier's signal
    # This guides the TF-IDF to extract strong tokens from the name patterns (e.g., M/T, M/V)
    df_out['fused_text_metadata'] = (
        "[NAME: " + clean_names + "] [LOG: " + clean_descs + "]"
    )
    
    # Clear out empty strings that don't contain real metadata
    # (i.e., rows that only have empty brackets left)
    has_metadata = df_out['fused_text_metadata'] != "[NAME: ] [LOG: ]"
    current_types = df_out['vessel_type'].astype(str).fillna('Not Recorded')
    
    # Establish strict masking for data-mining targets
    is_labeled = (current_types != 'Not Recorded') & has_metadata
    is_unlabeled = (current_types == 'Not Recorded') & has_metadata
    
    # Fallback guard clause in case memory arrays lack structural data split points
    if not is_labeled.any() or not is_unlabeled.any():
        return df_out

    # 1. Vectorization: Tokenize the fused corpus using unigrams and bigrams
    # sublinear_tf handles length variations between clean records and short logs
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=1500,  # Increased to accommodate name-based structural tokens
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_text = vectorizer.fit_transform(df_out.loc[is_labeled, 'fused_text_metadata'])
    y_train_labels = current_types[is_labeled]
    
    # 2. Linear Estimator: Fast convergence with balanced penalty adjustments
    nlp_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    nlp_model.fit(X_train_text, y_train_labels)
    
    # 3. Inference: Map target history gaps into the mathematical vector space
    X_unlabeled_text = vectorizer.transform(df_out.loc[is_unlabeled, 'fused_text_metadata'])
    predictions = nlp_model.predict(X_unlabeled_text)
    
    # 4. Imputation: Overwrite the 'Not Recorded' entries with structural inferences
    current_types.loc[is_unlabeled] = predictions
    
    # Re-align series memory into categorical schema blocks
    df_out['vessel_type'] = current_types.astype('category')
    
    # Drop the temporary fused feature to keep the dataframe's structure clean (DOD)
    df_out = df_out.drop(columns=['fused_text_metadata'])
    
    return df_out

def tx_discretize_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    
    # 1. Cria a flag binária de registro (Sinal para a árvore)
    df_out['feat_time_recorded'] = df_out['time'].notna().astype(np.int8)
    
    # 2. Agora sim, podemos deletar a coluna original de texto 'time' 
    # para não poluir o pré-processador
    df_out = df_out.drop(columns=['time'])
    
    return df_out



def tx_skewed_distance_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction: Transforms the heavily skewed 'shore_distance' feature.
    Applies a log1p transformation to distribute density evenly and creates 
    a structural binary flag for coastal/port opportunistic attacks.
    """
    df_out = df.copy()
    
    if 'shore_distance' in df_out.columns:
        # 1. Structural binary flag: Attacks happening extremely close to shore (e.g., ports/anchorage)
        # Tree models can use this indicator as an immediate entropy split point
        df_out['feat_is_near_port'] = (df_out['shore_distance'] <= 5.0).astype(np.int8)
        
        # 2. Logarithmic scale transformation (log1p handles 0 values gracefully: log(x + 1))
        # This converts the exponential wall into a smoother, bell-like distribution for linear layouts
        df_out['shore_distance_log'] = np.log1p(df_out['shore_distance'])
        
        # Drop the original skewed column to avoid multicollinearity and keep the memory block clean
        df_out = df_out.drop(columns=['shore_distance'])
        
    return df_out

def tx_handle_spatial_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing geospatial fields and extract maritime jurisdiction signals.
    
    Identifies high-seas incidents based on empty Exclusive Economic Zone (EEZ)
    entries and fills country metadata gaps with distinct semantic classes.
    """
    df_out = df.copy()
    
    if 'eez_country' in df_out.columns:
        # Create a structural binary flag for incidents in International Waters
        df_out['is_international_waters'] = df_out['eez_country'].isna().astype(np.int8)
        
        # Impute explicit semantic labels to fill remaining structural gaps
        df_out['eez_country'] = df_out['eez_country'].fillna('International Waters').astype('category')
        
    if 'nearest_country' in df_out.columns:
        df_out['nearest_country'] = df_out['nearest_country'].fillna('Unknown Coast').astype('category')
        
    return df_out

def tx_temporal_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Extract chronological anchors and compute cyclic wave-encoded time components.
    
    Transforms standard text dates into linear years for validation gating,
    and maps the day-of-year into sine/cosine dimensions to preserve seasonal
    patterns (e.g., monsoon cycles) for tree splitting.
    """
    df_out = df.copy()
    
    # Force transformation to a unified datetime block
    dates = pd.to_datetime(df_out['date'], errors='coerce')
    
    # 1. Structural Anchor (Mandatory for our TimeSeriesSplit tracking)
    df_out['year'] = dates.dt.year.fillna(2000).astype(np.int16)
    
    # 2. Cyclic Sseasonality Transformation
    # Day of year ranges from 1 to 365
    day_of_year = dates.dt.dayofyear.fillna(1)
    
    df_out['feat_season_sine'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_out['feat_season_cosine'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Drop the raw date string to prevent high-cardinality inflation
    df_out = df_out.drop(columns=['date'])
    
    return df_out

def tx_clean_vessels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise and clean vessel-related columns.

    Normalises vessel-type categories, drops duplicate IMO entries, and
    coerces date/time columns to ``datetime64`` where appropriate.
    """
    df_out = df.copy()
    
    # Clean administrative data sources to remove collection bias
    if 'data_source' in df_out.columns:
        df_out = df_out.drop(columns=['data_source'])
        
    # Standardise dynamic status mappings and preserve structural integrity
    if 'vessel_status' in df_out.columns:
        df_out['vessel_status'] = df_out['vessel_status'].astype(str).fillna('Unknown').astype('category')
        
    # Align target categories cleanly, dropping any dangling unknown target blocks
    if 'attack_type' in df_out.columns:
        df_out['attack_type'] = df_out['attack_type'].astype('category')
        
    return df_out

def tx_drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw text blocks, unencoded strings, and collinear spatial metadata.
    
    Removes high-cardinality descriptive identifiers and structural duplicates 
    to ensure the downstream matrix transformer receives a clean tabular grid.
    """
    df_out = df.copy()
    
    # Strict list of features to purge before exposing the schema to transformers
    features_to_drop = [
        'location_description',
        'attack_description',
        'vessel_name',
        'shore_longitude',
        'shore_latitude'
        'eez_country' # 0.93 NMI contra nearest_country
    ]
    
    # Drop only columns that actually exist in the dynamic dataframe frame
    existing_drops = [col for col in features_to_drop if col in df_out.columns]
    df_out = df_out.drop(columns=existing_drops)
    
    return df_out