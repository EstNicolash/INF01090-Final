import numpy as np
import pandas as pd
import pandas.api.types as pdt  # Pacote nativo para checagem estrita de tipos do Pandas
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def run_raw_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes a quick, transformations-free baseline benchmark.
    Fixes string-to-categorical constraints using native pandas API type checking.
    """
    df_sorted = df.sort_values('year').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['attack_type'])
    
    # 1. Naive Feature Selection: Drop unencoded unique texts and IDs immediately
    cols_to_drop = ['date', 'time', 'attack_description', 'vessel_name', 'location_description', 'data_source', 'attack_type']
    X_raw = df_sorted.drop(columns=[c for c in cols_to_drop if c in df_sorted.columns]).copy()
    
    # 2. Strict Categorical Mapping: Convert ALL non-numeric series to physical codes
    for col in X_raw.columns:
        # Usamos a API estável do Pandas que enxerga perfeitamente o <StringDtype>
        if not pdt.is_numeric_dtype(X_raw[col].dtype):
            X_raw[col] = X_raw[col].astype(str).astype('category').cat.codes
        else:
            # Standard fast numerical fill for remaining continuous missing blocks
            X_raw[col] = X_raw[col].fillna(-1)
        
    # Isolate Multi-class Target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_sorted['attack_type'].astype(str))
    num_classes = len(label_encoder.classes_)
    
    # Standard 3-Fold Temporal Split
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Instantiate models, adding the Zero-Intelligence Dummy baseline
    models = {
        "Dummy_Most_Frequent": DummyClassifier(strategy="most_frequent", random_state=42),
        "Random_Forest_Raw": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost_Raw": XGBClassifier(n_estimators=100, max_depth=4, eval_metric='mlogloss', random_state=42),
        "LightGBM_Raw": LGBMClassifier(n_estimators=100, max_depth=4, verbose=-1, random_state=42)
    }
    
    performance_records = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_raw)):
        X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Pad probability vectors if fold output lacks specific class blocks
            if y_proba.shape[1] < num_classes:
                completed_proba = np.zeros((y_proba.shape[0], num_classes))
                completed_proba[:, model.classes_] = y_proba
                y_proba = completed_proba
                
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            try:
                unique_test_classes = np.unique(y_test)
                if len(unique_test_classes) > 1:
                    auc_ovr = roc_auc_score(
                        y_test, 
                        y_proba[:, unique_test_classes] if y_proba.shape[1] > len(unique_test_classes) else y_proba, 
                        multi_class='ovr', 
                        average='macro',
                        labels=unique_test_classes
                    )
                else:
                    auc_ovr = 0.5
            except Exception:
                auc_ovr = 0.5
            
            performance_records.append({
                "Model": model_name,
                "Accuracy": acc,
                "F1_Macro": f1_macro,
                "AUC_ROC_OVR": auc_ovr
            })
            
    df_results = pd.DataFrame(performance_records)
    return df_results.groupby("Model").mean().reset_index()