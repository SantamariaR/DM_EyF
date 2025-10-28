# src/cleaning_features.py
# Módulo de funciones para la limpieza de features con canarios y otras técnicas. Usando Polars.
import lightgbm as lgb
import numpy as np
import logging
import polars as pl
from typing import List
from .config import MES_TRAIN, MES_VALIDACION, SEMILLA
from .gain_function import ganancia_evaluator_lgb

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



def train_overfit_lgbm_features(df: pl.DataFrame, objective: str = 'binary', num_class: int = None) -> lgb.Booster:
    
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
    periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    
    # ==================================================
    # Preparar dataset para LightGBM, entrenar y testear
    # ==================================================
    # Mapeo clase_ternaria a numérico
    mapping = {'CONTINUA': 0, 'BAJA+1': 1, 'BAJA+2': 1}
    
    X = df_train.drop(["clase_ternaria","numero_de_cliente"]).to_pandas()
    y = df_train["clase_ternaria"].to_pandas().map(mapping)

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
    
    return feature_importance_sorted





def seleccionar_variables_por_canaritos(feature_importance_sorted: dict, 
                                      df: pl.DataFrame, 
                                      porcentaje_umbral: float = 0.05) -> pl.DataFrame:
    """
    Selecciona variables importantes basándose en el porcentaje de canaritos que superan.
    
    Args:
        feature_importance_sorted: Diccionario con importancias de features (incluyendo canaritos)
        df: DataFrame original con todas las variables
        porcentaje_umbral: Porcentaje de canaritos que deben quedar por debajo (0.05 = 5%)
    
    Returns:
        DataFrame filtrado con solo las variables importantes
    """
    
    # Identificar canaritos
    canaritos = [feature for feature in feature_importance_sorted.keys() 
                if feature.startswith('canarito')]
    
    if not canaritos:
        logger.warning("⚠️ No se encontraron variables canarito en las importancias")
        return df
    
    logger.info(f"🔍 Encontrados {len(canaritos)} canaritos")
    
    # Obtener importancias de canaritos
    importancias_canaritos = [feature_importance_sorted[canarito] for canarito in canaritos]
    
    # Calcular percentil basado en el porcentaje umbral
    percentil_umbral = np.percentile(importancias_canaritos, (1 - porcentaje_umbral) * 100)
    
    logger.info(f"📊 Estadísticas de importancias de canaritos:")
    logger.info(f"   Mínimo: {min(importancias_canaritos):.6f}")
    logger.info(f"   Máximo: {max(importancias_canaritos):.6f}")
    logger.info(f"   Mediana: {np.median(importancias_canaritos):.6f}")
    logger.info(f"   Percentil {(1-porcentaje_umbral)*100:.1f}%: {percentil_umbral:.6f}")
    
    # Seleccionar variables que superan el umbral
    variables_importantes = []
    for feature, importancia in feature_importance_sorted.items():
        # Solo considerar variables que NO son canaritos y superan el umbral
        if not feature.startswith('canarito') and importancia > percentil_umbral:
            variables_importantes.append(feature)
    
    logger.info(f"✅ Variables seleccionadas: {len(variables_importantes)} de {len(feature_importance_sorted) - len(canaritos)}")
    
    # Agregar columnas obligatorias (identificadores y target)
    columnas_obligatorias = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']
    
    # Verificar que las columnas obligatorias existen en el DataFrame
    columnas_finales = []
    for col in columnas_obligatorias:
        if col in df.columns:
            columnas_finales.append(col)
        else:
            logger.warning(f"⚠️ Columna obligatoria '{col}' no encontrada en el DataFrame")
    
    # Combinar columnas obligatorias con variables importantes
    columnas_finales.extend(variables_importantes)
    
    # Filtrar DataFrame
    df_filtrado = df.select(columnas_finales)
    
    logger.info(f"📈 Dimensionalidad final: {df_filtrado.shape}")
    
    return df_filtrado