# src/cleaning_features.py
# Módulo de funciones para la limpieza de features con canarios y otras técnicas. Usando Polars.
import lightgbm as lgb
import numpy as np
import logging
import polars as pl
from typing import List
from .config import MES_TRAIN, MES_VALIDACION

logger = logging.getLogger(__name__)

def add_canaritos(df: pl.DataFrame, 
                          seed: int = 102191,
                          canaritos_ratio: float = 0.25,
                          columns_to_drop: List[str] = None) -> pl.DataFrame:
    """
    Réplica exacta del comportamiento de R que muestras:
    - Elimina columnas específicas
    - Añade columna 'azar' con valores uniformes
    - Ordena el DataFrame por 'azar' (manteniendo relaciones entre variables)
    - Elimina 'azar'
    - Añade features canaritos
    
    Args:
        df: DataFrame original de Polars
        seed: Semilla para reproducibilidad (102191 por defecto)
        canaritos_ratio: Proporción de features canaritos
        columns_to_drop: Columnas a eliminar antes de añadir canaritos
    
    Returns:
        DataFrame con canaritos y orden aleatorio
    """
    # Configurar semilla
    np.random.seed(seed)
    
    # Columnas por defecto a eliminar 
    if columns_to_drop is None:
        columns_to_drop = ['numero_de_cliente', 'clase_ternaria', 'foto_mes']
    
    # 1. Crear copia y eliminar columnas específicas
    df_canaritos = df.clone()
    
    # Filtrar solo las columnas que existen en el DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_canaritos.columns]
    if existing_columns_to_drop:
        df_canaritos = df_canaritos.drop(existing_columns_to_drop)
    
    # 2. Añadir columna 'azar' con distribución uniforme
    azar_values = np.random.uniform(0, 1, len(df_canaritos))
    df_canaritos = df_canaritos.with_columns(
        pl.Series('azar', azar_values)
    )
    
    # 3. Ordenar por 'azar' (mantiene relaciones entre variables)
    df_canaritos = df_canaritos.sort('azar')
    
    # 4. Eliminar columna 'azar'
    df_canaritos = df_canaritos.drop('azar')
    
    # 5. Calcular número de canaritos a añadir
    n_original_features = len(df_canaritos.columns)
    n_canaritos = int(n_original_features * canaritos_ratio)
    
    if n_canaritos == 0:
        n_canaritos = 1
    
    # 6. Añadir features canaritos
    for i in range(n_canaritos):
        # Generar datos aleatorios normales
        random_data = np.random.normal(0, 1, len(df_canaritos))
        canarito_name = f'canarito_{i:03d}'
        
        df_canaritos = df_canaritos.with_columns(
            pl.Series(canarito_name, random_data)
        )
    
    print(f"✅ Proceso completado con semilla: {seed}")
    print(f"📊 Columnas eliminadas: {existing_columns_to_drop}")
    print(f"🎯 Features originales: {n_original_features}")
    print(f"🔔 Features canaritos añadidas: {n_canaritos}")
    print(f"📈 Features totales: {len(df_canaritos.columns)}")
    
    return df_canaritos



def train_overfit_lgbm(df: pl.DataFrame, objective: str = 'multiclass', num_class: int = 3) -> lgb.Booster:
    
    """
    Entrenamos un modelo lightGBM hasta el sobreajuste total para detectar features inútiles (canaritos).
    Args:
        df_train: DataFrame de Polars con datos de entrenamiento (incluye target)
        objective: Objetivo de LightGBM (por defecto 'multiclass')
        num_class: Número de clases (por defecto 3)
    Returns:
        Modelo LightGBM entrenado hasta sobreajuste
    """
    params = {
        'objective': objective,
        'metric': 'multi_logloss',
        'num_class': num_class,
        'boosting_type': 'gbdt',
        
       
        'num_leaves': 2**10,          
        'max_depth': -1,               
        'min_data_in_leaf': 1,       
        'min_child_samples': 1,      
        'min_split_gain': 0.0,         
        
    
        'lambda_l1': 0.0,             
        'lambda_l2': 0.0,             
        

        'feature_fraction': 1.0,       
        'bagging_fraction': 1.0,    
        'bagging_freq': 0,             
        'learning_rate': 0.1,
        'verbosity': -100
    }

    
    # Períodos de evaluación
    periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    
    # ==================================================
    # Preparar dataset para LightGBM, entrenar y testear
    # ==================================================
    X = df_train.drop(["clase_ternaria"]).to_pandas()
    y = df_train["clase_ternaria"].to_pandas()

    dtrain = lgb.Dataset(X, label=y)
    
    modelo = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    return modelo

