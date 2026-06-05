import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna

# Importações do Scikit-Learn e Modelos
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Garantir o caminho do projeto no Python Path se rodar fora da raiz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.pipelines import build_preprocessing_pipeline

# Desativar warnings poluentes durante os trials
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_hyperparameters(model_instance, X, y, tscv, preprocessor) -> float:
    """
    Roda a validação temporal estrita e calcula a média do F1-Macro 
    para o conjunto específico de hiperparâmetros.
    """
    f1_folds = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Transformação isolada por fold (Sem Data Leakage)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        # Ajuste fino: Se o modelo for do LightGBM/XGBoost e faltar classe no treino,
        # esses algoritmos podem reclamar. Treinamos normalmente.
        model_instance.fit(X_train_trans, y_train)
        y_pred = model_instance.predict(X_test_trans)
        
        # Métrica de otimização focada em minoritárias
        fold_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_folds.append(fold_f1)
        
    return float(np.mean(f1_folds))

def run_optuna_tuning(df: pd.DataFrame, n_trials: int = 50):
    """
    Configura e orquestra os estudos independentes do Optuna para cada modelo.
    """
    print("====== Preparando Matrizes para o Tuning Temporal ======")
    df_sorted = df.sort_values('year').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['attack_type'])
    
    X = df_sorted.drop(columns=['attack_type'])
    le = LabelEncoder()
    y = le.fit_transform(df_sorted['attack_type'].astype(str))
    num_classes = len(le.classes_)
    
    tscv = TimeSeriesSplit(n_splits=3)
    preprocessor = build_preprocessing_pipeline()
    
    # -------------------------------------------------------------------------
    # OBJECTIVE: LIGHTGBM
    # -------------------------------------------------------------------------
    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        return evaluate_hyperparameters(model, X, y, tscv, preprocessor)

    # -------------------------------------------------------------------------
    # OBJECTIVE: XGBOOST
    # -------------------------------------------------------------------------
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        return evaluate_hyperparameters(model, X, y, tscv, preprocessor)

    # -------------------------------------------------------------------------
    # OBJECTIVE: RANDOM FOREST
    # -------------------------------------------------------------------------
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        return evaluate_hyperparameters(model, X, y, tscv, preprocessor)

    # Execução dos Estudos usando TPESampler (Otimizador Bayesiano nativo)
    models_to_tune = {
        "LightGBM": objective_lgb,
        "XGBoost": objective_xgb,
        "Random_Forest": objective_rf
    }
    
    best_parameters_repo = {}
    
    for model_name, objective_func in models_to_tune.items():
        print(f"\n🚀 Disparando Tuning Bayesiano para o {model_name} ({n_trials} trials)...")
        
        # Armazenamento local em banco SQLite para segurança dos dados coletados
        db_path = f"sqlite:///optuna_study_{model_name.lower()}.db"
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"tuning_{model_name.lower()}",
            storage=db_path,
            load_if_exists=True
        )
        
        study.optimize(objective_func, n_trials=n_trials, n_jobs=1)
        
        print(f"✅ Finalizado! Melhor F1-Macro atingido para {model_name}: {study.best_value:.4f}")
        best_parameters_repo[model_name] = study.best_params

    # Print final limpo estruturado em dicionários prontos para cópia
    print("\n" + "="*80)
    print("      DICIONÁRIOS DE HIPERPARÂMETROS OTIMIZADOS (COPIE ABAIXO)")
    print("="*80)
    
    for model_name, best_params in best_parameters_repo.items():
        print(f"\n# Hiperparâmetros recomendados para injeção no {model_name}:")
        print(f"best_params_{model_name.lower()} = {best_params}")

if __name__ == "__main__":
    # Script para ser executado via terminal apontando para os seus dados tratados
    # Exemplo: python src/models/tuning.py
    print("Carregando df_processed da memória ou de arquivo persistido...")
    # Modifique o caminho de leitura conforme o salvamento do seu notebook anterior
    try:
        df_processed = pd.read_parquet("data/processed/df_processed.parquet")
        run_optuna_tuning(df_processed, n_trials=30)
    except FileNotFoundError:
        print("❌ Erro: Salve o seu 'df_processed' em 'data/processed/df_processed.parquet' no notebook antes de rodar o script.")