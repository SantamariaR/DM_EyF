# src/cleaning_features.py
# Módulo de funciones para la limpieza de features con canarios y otras técnicas. Usando Polars.
import lightgbm as lgb
import numpy as np
import logging
import polars as pl
from typing import List
from .config import MES_TRAIN, MES_VALIDACION, SEMILLA, BUCKET_NAME, STUDY_NAME
from .gain_function import ganancia_evaluator_lgb
import os
import pandas as pd

logger = logging.getLogger(__name__)

def add_canaritos(df: pl.DataFrame, 
                          seed: int = 102191,
                          canaritos_ratio: float = 0.5,
                          columns_to_ignore: List[str] = None) -> pl.DataFrame:
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
    
    # Columnas por defecto a ignorar
    if columns_to_ignore is None:
        columns_to_ignore = ['numero_de_cliente', 'clase_ternaria', 'foto_mes']
    
    # 1. Crear copia y añadir columna 'azar'
    azar_values = np.random.uniform(0, 1, len(df))
    df_canaritos = df.with_columns(
        pl.Series('azar', azar_values)
    )
    
    # 2. Ordenar por 'azar' (mantiene relaciones entre variables)
    df_canaritos = df_canaritos.sort('azar')
    
    # 3. Calcular número de canaritos basado en columnas NO ignoradas
    # Excluimos 'azar' y las columnas a ignorar del cálculo
    columns_for_canaritos_count = [col for col in df.columns 
                                   if col not in columns_to_ignore]
    n_original_features = len(columns_for_canaritos_count)
    n_canaritos = max(1, int(n_original_features * canaritos_ratio))
    
    # 4. Añadir features canaritos al DataFrame ordenado
    for i in range(n_canaritos):
        random_data = np.random.normal(0, 1, len(df_canaritos))
        canarito_name = f'canarito_{i:03d}'
        
        df_canaritos = df_canaritos.with_columns(
            pl.Series(canarito_name, random_data)
        )
    
    # 5. Eliminar columna 'azar' (ya cumplió su función de ordenar)
    df_canaritos = df_canaritos.drop('azar')
    
    print(f"✅ Proceso completado con semilla: {seed}")
    print(f"📊 Columnas ignoradas: {columns_to_ignore}")
    print(f"🎯 Features originales (para cálculo): {n_original_features}")
    print(f"🔔 Features canaritos añadidas: {n_canaritos}")
    print(f"📈 Features totales: {len(df_canaritos.columns)}")
    
    return df_canaritos



def train_overfit_lgbm_features(df: pl.DataFrame, objective: str = 'binary', undersampling: float = 0.5,archivo_base=None) -> lgb.Booster:
    
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
        #'metric': 'binary_logloss',
        #'num_class': num_class,
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
    mitad = (len(MES_TRAIN) + 1) // 2
    periodos_entrenamiento = MES_TRAIN[:mitad]
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
        
    # Separar clases
    df_pos = df_train.filter(pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"]))
    df_neg = df_train.filter(pl.col("clase_ternaria") == "CONTINUA")

    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
    n_sample = int(df_neg.height * undersampling)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_sub = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0])
    # ==================================================
    # Preparar dataset para LightGBM, entrenar y testear
    # ==================================================
    # Mapeo clase_ternaria a numérico
    mapping = {'CONTINUA': 0, 'BAJA+1': 1, 'BAJA+2': 1}
    
    X = df_sub.drop(["clase_ternaria","numero_de_cliente"]).to_pandas()
    y = df_sub["clase_ternaria"].to_pandas().map(mapping)

    dtrain = lgb.Dataset(X, label=y)
    
    # Diccionario para acumular importancias
    feature_importance_total = {}
    
    #modelos = []
    
    # Entrenar 5 modelos con diferentes semillas
    for i, semilla in enumerate(SEMILLA):
        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
        
        # Actualizar la semilla en los parámetros
        params['seed'] = semilla
        params['feature_fraction_seed'] = semilla
        params['bagging_seed'] = semilla
        
        # Entrenar modelo
        model = lgb.train(
            params,
            dtrain,
            feval=ganancia_evaluator_lgb,
            num_boost_round=100            
        )
        #modelos.append(model)
        # Obtener importancia de features
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        # Acumular importancias
        for feature, imp in zip(feature_names, importance):
            if feature not in feature_importance_total:
                feature_importance_total[feature] = 0
            feature_importance_total[feature] += imp
        
        print(f"✅ Modelo {i+1}/{5} completado (seed: {semilla})")
        
    # Calcular importancia promedio
    feature_importance_avg = {k: v / 5 for k, v in feature_importance_total.items()}
    
    # Ordenar por importancia
    feature_importance_sorted = dict(sorted(
        feature_importance_avg.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    print(f"\n📊 Resumen de importancia de features ({5} modelos):")
    print(f"🔝 Top 10 features más importantes:")
    for i, (feature, importance) in enumerate(list(feature_importance_sorted.items())[:10]):
        print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    print(f"\n🔚 Bottom 10 features menos importantes:")
    for i, (feature, importance) in enumerate(list(feature_importance_sorted.items())[-10:]):
        print(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "exp")
    os.makedirs(path_db, exist_ok=True)
    
    # Ruta completa de la base de datos
    archivo_csv = os.path.join(path_db, f"{archivo_base}_feature_importance.csv")
    
    # ✅ FORMA CORRECTA: Crear DataFrame con dos columnas
    df_feature_importance = pd.DataFrame(
        list(feature_importance_sorted.items()),  # Convertir a lista de tuplas
        columns=['Feature', 'Importance']         # Nombres de columnas
    )
    
    # ✅ FORMA ALTERNATIVA: Desde el diccionario original
    # df_feature_importance = pd.DataFrame([
    #     {'Feature': k, 'Importance': v} 
    #     for k, v in feature_importance_sorted.items()
    # ])
    
    df_feature_importance.to_csv(archivo_csv, index=False)
    logger.info(f"✅ Feature importance guardado en: {archivo_csv}")
    return feature_importance_sorted



def seleccionar_variables_por_canaritos(feature_importance_sorted: dict, 
                                      df: pl.DataFrame, 
                                      porcentaje_umbral: float = 0.05) -> pl.DataFrame:
    """
    Versión optimizada que evita duplicados usando conjuntos
    """
    # Identificar canaritos
    canaritos = [feature for feature in feature_importance_sorted.keys() 
                if feature.startswith('canarito')]
    
    if not canaritos:
        logger.warning("⚠️ No se encontraron variables canarito en las importancias")
        return df
    
    # Calcular umbral
    importancias_canaritos = [feature_importance_sorted[canarito] for canarito in canaritos]
    percentil_umbral = np.percentile(importancias_canaritos, (1 - porcentaje_umbral) * 100)
    
    logger.info(f"📊 Umbral de selección: {percentil_umbral:.6f}")
    
    # Seleccionar variables importantes (excluyendo canaritos)
    variables_importantes = {
        feature for feature, importancia in feature_importance_sorted.items()
        if not feature.startswith('canarito') and importancia > percentil_umbral
    }
    
    # Columnas obligatorias
    columnas_obligatorias = {'numero_de_cliente', 'foto_mes', 'clase_ternaria'}
    
    # Combinar sin duplicados
    columnas_finales = columnas_obligatorias.union(variables_importantes)
    
    # Filtrar solo las que existen en el DataFrame
    columnas_existentes = set(df.columns)
    columnas_finales = list(columnas_finales.intersection(columnas_existentes))
    
    # Ordenar: primero obligatorias, luego alfabético
    columnas_ordenadas = []
    for col in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']:
        if col in columnas_finales:
            columnas_ordenadas.append(col)
    
    # Resto de columnas ordenadas
    otras_columnas = sorted([col for col in columnas_finales if col not in columnas_ordenadas])
    columnas_ordenadas.extend(otras_columnas)
    
    logger.info(f"✅ {len(variables_importantes)} variables seleccionadas")
    logger.info(f"📋 {len(columnas_ordenadas)} columnas finales")
    logger.info(f"📈 Dimensionalidad: {df.shape} → {[df.shape[0], len(columnas_ordenadas)]}")
    
    return df.select(columnas_ordenadas)