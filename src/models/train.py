import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.models.pipelines import build_preprocessing_pipeline

def instantiate_models(num_classes: int) -> dict:
    """Instantiates the multi-model benchmarking dictionary with explicit objectives."""
    return {
        "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, objective="multi:softprob", eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42, verbose=-1),
        "Neural_Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, activation='relu', random_state=42)
    }

def run_temporal_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates sequential training and evaluation using a TimeSeriesSplit loop.
    Ensures zero state leakage between historical folds and safe multi-class metrics.
    """
    df_sorted = df.sort_values('year').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['attack_type'])
    
    X = df_sorted.drop(columns=['attack_type'])
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_sorted['attack_type'].astype(str))
    num_classes = len(label_encoder.classes_)
    
    tscv = TimeSeriesSplit(n_splits=3)
    preprocessor = build_preprocessing_pipeline()
    
    performance_records = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # AJUSTE 1: Instanciar novos modelos do zero a cada fold para evitar leak cumulativo de pesos
        models = instantiate_models(num_classes)
        
        # Fit and transform features linearly for this iteration's frame
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        for model_name, model in models.items():
            model.fit(X_train_trans, y_train)
            
            y_pred = model.predict(X_test_trans)
            y_proba = model.predict_proba(X_test_trans)
            
            if y_proba.shape[1] < num_classes:
                completed_proba = np.zeros((y_proba.shape[0], num_classes))
                completed_proba[:, model.classes_] = y_proba
                y_proba = completed_proba

            acc = accuracy_score(y_test, y_pred)
            
            prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            prec_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
            rec_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
            f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
            
            # AJUSTE 2: Cálculo resiliente do AUC-ROC para evitar colapso por falta de classes no bloco temporal
            try:
                unique_test_classes = np.unique(y_test)
                if len(unique_test_classes) > 1:
                    # Filtra as probabilidades apenas para as classes presentes no teste atual
                    # e re-normaliza o vetor para que a soma das linhas continue sendo 1.0
                    proba_filtered = y_proba[:, unique_test_classes]
                    proba_filtered = proba_filtered / proba_filtered.sum(axis=1, keepdims=True)
                    
                    auc_ovr = roc_auc_score(
                        y_test, 
                        proba_filtered, 
                        multi_class='ovr', 
                        average='macro',
                        labels=unique_test_classes
                    )
                else:
                    auc_ovr = 0.5
            except Exception:
                auc_ovr = 0.5
            
            performance_records.append({
                "Fold": fold + 1,
                "Model": model_name,
                "Accuracy": acc,
                "F1_Macro": f1_macro,
                "Precision_Macro": prec_macro,
                "Recall_Macro": rec_macro,
                "F1_Micro": f1_micro,
                "Precision_Micro": prec_micro,
                "Recall_Micro": rec_micro,
                "AUC_ROC_OVR": auc_ovr
            })
            
    df_results = pd.DataFrame(performance_records)
    df_summary = df_results.groupby("Model").mean().drop(columns=["Fold"]).reset_index()
    return df_summary
