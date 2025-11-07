import optuna
import lightgbm as lgb
import math
import polars as pl
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_evaluator_lgb,ganancia_evaluator_manual, calcular_ganancia_acumulada

logger = logging.getLogger(__name__)


def objetivo_ganancia(trial: optuna.trial.Trial, df: pl.DataFrame, undersampling: float = 1) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
  
    periodos_entrenamiento = MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN]
    periodo_validacion = MES_VALIDACION
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de validación: {periodo_validacion}")
        
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))

    # Separar clases
    df_pos = df_train.filter(pl.col("clase_01") == 1)
    df_neg = df_train.filter(pl.col("clase_01") == 0)

    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
    n_sample = int(df_neg.height * undersampling)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0] + trial.number)

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_sub = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0])
    
    # Preparar datos de validación
    df_val = df.filter(pl.col("foto_mes").is_in(periodo_validacion))

    # ===============================
    # Preparar dataset para LightGBM
    # ===============================
    X = df_sub.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y = df_sub["clase_01"].to_pandas()

    dtrain = lgb.Dataset(X, label=y)
    
    X_val = df_val.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_val = df_val["clase_01"].to_pandas()
    
    val_data = lgb.Dataset(X_val, label=y_val)
    
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()}")
   
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_val = []
    
    # HIPERPARÁMETROS CON OPTIMIZACIÓN BAYESIANA MEJORADA
    cant_registros = X.shape[0]
    
    # Parámetros con transformaciones similares al código R original
    num_iterations_exp = trial.suggest_float('num_iterations_exp', 0.0, 11.1)
    num_iterations = int(round(2 ** num_iterations_exp))
    
    learning_rate_exp = trial.suggest_float('learning_rate_exp', -8.0, -1.0)
    learning_rate = 2 ** learning_rate_exp
    
    feature_fraction = trial.suggest_float('feature_fraction', 0.05, 1.0)
    
    # min_data_in_leaf con límite superior dinámico
    max_min_data_exp = math.log2(cant_registros / 2)
    min_data_exp = trial.suggest_float('min_data_exp', 0.0, max_min_data_exp)
    min_data_in_leaf = int(round(2 ** min_data_exp))
    
    num_leaves_exp = trial.suggest_float('num_leaves_exp', 1.0, 10.0)
    num_leaves = int(round(2 ** num_leaves_exp))
    
    # Restricción forbidden del código R original
    if min_data_in_leaf * num_leaves > cant_registros:
        logger.info(f"Restricción violada: {min_data_in_leaf} * {num_leaves} > {cant_registros}")
        return -1000000  # Penalización fuerte
    
    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "None",
        "first_metric_only": False,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        
        "seed": 143287,
        "feature_fraction_seed": 143287,
        'bagging_seed': 143287,
        "max_depth": -1,
        "min_gain_to_split": 0,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 31,
        
        "bagging_fraction": 1.0,
        "pos_bagging_fraction": 1.0,
        "neg_bagging_fraction": 1.0,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,
        
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        
        "extra_trees": False,
        
        # PARÁMETROS OPTIMIZADOS
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "feature_fraction": feature_fraction,
        "min_data_in_leaf": min_data_in_leaf,
        "num_leaves": num_leaves
    }
    
    logger.info(f"Parámetros del trial: iteraciones={num_iterations}, lr={learning_rate:.6f}, "
                f"feature_frac={feature_fraction:.3f}, min_data={min_data_in_leaf}, leaves={num_leaves}")
    
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
            valid_sets=[val_data],
            feval=ganancia_evaluator_lgb,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Guardar modelo
        modelos.append(model)
        
        # Predecir con este modelo
        y_pred_proba_single = abs(model.predict(X_val))
        predicciones_val.append(y_pred_proba_single)

        # Calcular ganancia individual
        ganancia_individual = ganancia_evaluator_manual(y_val, y_pred_proba_single)  
        logger.info(f"Ganancia modelo {i+1}: {ganancia_individual}")

    # Promediar las predicciones de validación
    y_pred_proba_ensemble = np.mean(predicciones_val, axis=0)
    
    # Calcular ganancia del ensemble
    ganancia_ensemble = ganancia_evaluator_manual(y_val, y_pred_proba_ensemble)
    logger.info(f"\n=== GANANCIA ENSEMBLE: {ganancia_ensemble} ===")    

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_ensemble)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_ensemble:,.0f}")
  
    return ganancia_ensemble



#def objetivo_ganancia(trial: optuna.trial.Trial, df: pl.DataFrame, undersampling: float = 1) -> float:
#    """
#    Parameters:
#    trial: trial de optuna
#    df: dataframe con datos
#  
#    Description:
#    Función objetivo que maximiza ganancia en mes de validación.
#    Utiliza configuración YAML para períodos y semilla.
#    Define parametros para el modelo LightGBM
#    Preparar dataset para entrenamiento y validación
#    Entrena modelo con función de ganancia personalizada
#    Predecir y calcular ganancia
#    Guardar cada iteración en JSON
#  
#    Returns:
#    float: ganancia total
#    """
#  
#    # Usar target (clase_ternaria ya convertida a binaria)
#  
#    # Features: usar todas las columnas excepto target
#  
#    # Entrenar modelo con función de ganancia personalizada
#    
#    periodos_entrenamiento = MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN]
#    periodo_validacion = MES_VALIDACION
#        
#    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
#    logger.info(f"Período de validación: {periodo_validacion}")
#        
#    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
#
#    # Separar clases
#    df_pos = df_train.filter(pl.col("clase_01") == 1)
#    df_neg = df_train.filter(pl.col("clase_01") == 0)
#
#
#    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
#    n_sample = int(df_neg.height * undersampling)
#    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0] + trial.number)
#
#    # Concatenar positivos y negativos muestreados
#    df_sub = pl.concat([df_pos, df_neg])
#
#    # Shuffle del dataset
#    df_sub = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0])
#    
#    # Preparar datos de validación
#    df_val = df.filter(pl.col("foto_mes").is_in(periodo_validacion))
#
#
#    # ===============================
#    # Preparar dataset para LightGBM
#    # ===============================
#    X = df_sub.drop(["clase_ternaria", "clase_01"]).to_pandas()
#    y = df_sub["clase_01"].to_pandas()
#
#    dtrain = lgb.Dataset(X, label=y)
#    
#    X_val = df_val.drop(["clase_ternaria", "clase_01"]).to_pandas()
#    y_val = df_val["clase_01"].to_pandas()
#    
#    val_data = lgb.Dataset(X_val, label=y_val)
#    
#    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
#   
#    # Listas para almacenar modelos y predicciones
#    modelos = []
#    predicciones_val = []
#    # Hiperparámetros a optimizar
#
#    cant_registros = X.shape[0]    
#      
#    params = {
#      "boosting": "gbdt", # puede ir  dart  , ni pruebe random_forest
#      "objective": "binary",
#      "metric": "None",
#      "first_metric_only": False,
#      "boost_from_average": True,
#      "feature_pre_filter": False,
#      "force_row_wise": True, # para reducir warnings
#      "verbosity": -100,
#    
#      "seed": 143287,
#      "feature_fraction_seed": 143287,
#      'bagging_seed': 143287,
#      "max_depth": -1, # -1 significa no limitar,  por ahora lo dejo fijo
#      "min_gain_to_split": 0, # min_gain_to_split >= 0
#      "min_sum_hessian_in_leaf": 0.001, #  min_sum_hessian_in_leaf >= 0.0
#      "lambda_l1": 0.0, # lambda_l1 >= 0.0
#      "lambda_l2": 0.0, # lambda_l2 >= 0.0
#      "max_bin": 31, # lo debo dejar fijo, no participa de la BO
#    
#      "bagging_fraction": 1.0, # 0.0 < bagging_fraction <= 1.0
#      "pos_bagging_fraction": 1.0, # 0.0 < pos_bagging_fraction <= 1.0
#      "neg_bagging_fraction": 1.0,# 0.0 < neg_bagging_fraction <= 1.0
#      "is_unbalance": False, #
#      "scale_pos_weight": 1.0, # scale_pos_weight > 0.0
#    
#      "drop_rate": 0.1, # 0.0 < neg_bagging_fraction <= 1.0
#      "max_drop": 50, # <=0 means no limit
#      "skip_drop": 0.5, # 0.0 <= skip_drop <= 1.0
#    
#      "extra_trees": False,
#    
#      "num_iterations": trial.suggest_int('num_iterations',2048 ,4096 ),
#      "learning_rate": trial.suggest_float('learning_rate', 0.002,0.8 ),
#      "feature_fraction": trial.suggest_float('feature_fraction',0.2 , 0.8),
#      "min_data_in_leaf" : trial.suggest_int('min_data_in_leaf',1 ,2048),
#      "num_leaves": trial.suggest_int('num_leaves', 2, 82)  
#    }
#    
#    # Entrenar 5 modelos con diferentes semillas
#    for i, semilla in enumerate(SEMILLA):
#        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
#        
#        # Actualizar la semilla en los parámetros
#        params['seed'] = semilla
#        params['feature_fraction_seed'] = semilla
#        params['bagging_seed'] = semilla
#        
#        # Entrenar modelo
#        model = lgb.train(
#            params,
#            dtrain,
#            valid_sets=[val_data],
#            feval=ganancia_evaluator_lgb,
#            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
#        )
#        
#        # Guardar modelo
#        modelos.append(model)
#        
#        # Predecir con este modelo
#        y_pred_proba_single = abs(model.predict(X_val))
#        predicciones_val.append(y_pred_proba_single)
#
#        
#        # Calcular ganancia individual
#        ganancia_individual = ganancia_evaluator_manual(y_val,y_pred_proba_single)  
#        logger.info(f"Ganancia modelo {i+1}: {ganancia_individual}")
#
#    
#    
#    # Promediar las predicciones de validación
#    y_pred_proba_ensemble = np.mean(predicciones_val, axis=0)
#    
#   
#    # Calcular ganancia del ensemble
#    ganancia_ensemble = ganancia_evaluator_manual(y_val, y_pred_proba_ensemble)
#    logger.info(f"\n=== GANANCIA ENSEMBLE: {ganancia_ensemble} ===")    
#
#    # Guardar cada iteración en JSON
#    guardar_iteracion(trial, ganancia_ensemble)
#  
#    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_ensemble:,.0f}")
#  
#    return ganancia_ensemble


#def optimizar(df: pl.DataFrame, n_trials: int = 1) -> optuna.Study:
#    """
#    Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
#    """
#
#    study_name = STUDY_NAME
#
#    logger.info(f"Iniciando optimización ")
#    logger.info(f"Configuración: períodos Entrenamiento={MES_TRAIN }, Validación={MES_VALIDACION}")
#
#     # Crear estudio
#    study = optuna.create_study(
#        direction='maximize',
#        study_name=study_name,
#        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
#    )
#  
#    # Ejecutar optimización
#    study.optimize(lambda trial: objetivo_ganancia(trial, df), n_trials=n_trials)
#
##    # Normalización de parámetros
##    update_dict = {
##        'min_data_in_leaf': round(study.best_params['min_data_in_leaf'] / HIPERPARAM_BO['UNDERSUMPLING'])
##    }
#    #study.best_params.update(update_dict)
#
#    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
#    logger.info(f"Mejores parámetros: {study.best_params}")
#
#    return study


def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
        
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "exp")
    os.makedirs(path_db, exist_ok=True)
  
    # Nombre del archivo único para todas las iteraciones
    archivo = os.path.join(path_db, f"{archivo_base}_iteraciones.json")
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")



   
def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    
        # Hiperparámetros a optimizar
    params = {
      "boosting": "gbdt", # puede ir  dart  , ni pruebe random_forest
      "objective": "binary",
      "metric": "None",
      "first_metric_only": False,
      "boost_from_average": True,
      "feature_pre_filter": False,
      "force_row_wise": True, # para reducir warnings
      "verbosity": -100,
    
      "seed": 143287,
      "feature_fraction_seed": 143287,
      'bagging_seed': 143287,
      "max_depth": -1, # -1 significa no limitar,  por ahora lo dejo fijo
      "min_gain_to_split": 0, # min_gain_to_split >= 0
      "min_sum_hessian_in_leaf": 0.001, #  min_sum_hessian_in_leaf >= 0.0
      "lambda_l1": 0.0, # lambda_l1 >= 0.0
      "lambda_l2": 0.0, # lambda_l2 >= 0.0
      "max_bin": 31, # lo debo dejar fijo, no participa de la BO
    
      "bagging_fraction": 1.0, # 0.0 < bagging_fraction <= 1.0
      "pos_bagging_fraction": 1.0, # 0.0 < pos_bagging_fraction <= 1.0
      "neg_bagging_fraction": 1.0,# 0.0 < neg_bagging_fraction <= 1.0
      "is_unbalance": False, #
      "scale_pos_weight": 1.0, # scale_pos_weight > 0.0
    
      "drop_rate": 0.1, # 0.0 < neg_bagging_fraction <= 1.0
      "max_drop": 50, # <=0 means no limit
      "skip_drop": 0.5, # 0.0 <= skip_drop <= 1.0
    
      "extra_trees": False,
    
      "num_iterations": 0,
      "learning_rate": 0,
      "feature_fraction": 0,
      "num_leaves": 0,
      "min_data_in_leaf" : 0  
    }
    
    # Actualizar parámetros con los sugeridos por Optuna
    params.update(mejores_params)
   
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Períodos de evaliación
    periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
    #periodo_validacion = MES_VALIDACION
    periodo_test = MES_TEST
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    #logger.info(f"Período de Validación: {periodo_validacion}")
    logger.info(f"Período de Testeo: {periodo_test}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    #df_val = df.filter(pl.col("foto_mes").is_in(periodo_validacion))
    df_test = df.filter(pl.col("foto_mes").is_in(periodo_test))
    
    # Separar clases(por si queremos undersampling )
    df_pos = df_train.filter(pl.col("clase_01") == 1)
    df_neg = df_train.filter(pl.col("clase_01") == 0)

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_sub = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0]) 
    

    # ==================================================
    # Preparar dataset para LightGBM, entrenar y testear
    # ==================================================
    X = df_sub.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y = df_sub["clase_01"].to_pandas()

    dtrain = lgb.Dataset(X, label=y)
    
    #X_val = df_val.drop(["clase_ternaria", "clase_01"]).to_pandas()
    #y_val = df_val["clase_01"].to_pandas()
    
    #val_data = lgb.Dataset(X_val, label=y_val) 
    
    # Para test sólo tomo como positivos BAJA+2
    X_test = df_test.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_test = df_test["clase_ternaria"].to_pandas()
    y_test = (y_test == "BAJA+3").astype(int)
    
      
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
   
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_test = []
    # Entrenar 30 modelos con diferentes semillas
    for i, semilla in enumerate(SEMILLA+SEMILLERO):
        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
        
        # Actualizar la semilla en los parámetros
        params['seed'] = semilla
        params['feature_fraction_seed'] = semilla
        params['bagging_seed'] = semilla
        
        # Entrenar modelo
#        model = lgb.train(
#            params,
#            dtrain,
#            feval=ganancia_evaluator_lgb,
#            valid_sets=[val_data],
#            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0),
#            lgb.log_evaluation(period=100)]
#            
#        )
        model = lgb.train(
            params,
            dtrain,
            feval=ganancia_evaluator_lgb,
            num_boost_round=100,  # Especifica el número de iteraciones directamente
            callbacks=[lgb.log_evaluation(period=100)]
        )        
        # Guardar modelo
        modelos.append(model)
        
        # Predecir con este modelo
        y_pred_proba_single = abs(model.predict(X_test))
        logger.info(f"Predicciones modelo {i+1} (primeros 10): {y_pred_proba_single[:10]}")
        predicciones_test.append(y_pred_proba_single)    

    # Promediar las predicciones de test
    y_pred_proba_ensemble = np.mean(predicciones_test, axis=0)
    
    # Le añado la columna de predicciones al df_test original para análisis posterior si es necesario
    df_test = df_test.with_columns([
        pl.Series("pred_proba_ensemble", y_pred_proba_ensemble)
    ])
    
    #logger.info(f"Datos de test con predicciones listos: {df_test.head()}")
    # Calcular solo la ganancia
    
    # Calculamos las ganancias para distintos cortes
    df_pred = calcular_ganancia_acumulada(df_test, col_probabilidad="pred_proba_ensemble", col_clase="clase_ternaria")
  
 
    return  df_pred

def guardar_resultados_test(df_resultado, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo CSV.
    
    Args:
        df_resultado: DataFrame de Polars con la curva de ganancia
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME

    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "exp")
    os.makedirs(path_db, exist_ok=True)
  
    # Nombre del archivo único para todas las iteraciones
    archivo_csv = os.path.join(path_db, f"{archivo_base}_test_mes_{MES_TEST}.csv")
    
   
    # Guardar directamente como CSV desde Polars
    df_resultado.write_csv(archivo_csv)
    
    # Me creo un sub de entre 5000 y 30000
    df_reducido = df_resultado.filter(
        (pl.col('cantidad_clientes') >= 0) & (pl.col('cantidad_clientes') <= 30000)
    )
    
    
    # Calcular métricas para el log
    ganancia_maxima = df_reducido['ganancia_acumulada'].max()
    cantidad_optima = df_reducido.filter(
        pl.col('ganancia_acumulada') == ganancia_maxima
    )['cantidad_clientes'].min()
    
    logger.info(f"Testeo del MES {MES_TEST} guardado en {archivo_csv}")
    logger.info(f"Ganancia máxima: {ganancia_maxima:,.0f} con {cantidad_optima} envíos")
    logger.info(f"Total de registros: {len(df_resultado):,}")



def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
    
    
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            #PRESTAR ATENCION Y RAZONAR!!!
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
     # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )

  
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study


def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME
    

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        ##LO UNICO IMPORTANTE DEL METODO Y EL study CLARO
        study.optimize(lambda trial: objetivo_ganancia(trial, df, undersampling), n_trials=trials_a_ejecutar)
        
        
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
        
    # Obtener y mostrar parámetros transformados
    best_trial = study.best_trial
    
    best_params_transformed = {
        'num_iterations': int(round(2 ** best_trial.params['num_iterations_exp'])),
        'learning_rate': 2 ** best_trial.params['learning_rate_exp'],
        'feature_fraction': best_trial.params['feature_fraction'],
        'min_data_in_leaf': int(round(2 ** best_trial.params['min_data_exp'])),
        'num_leaves': int(round(2 ** best_trial.params['num_leaves_exp']))
    }
    
    # Agregar como atributo custom al estudio
    study.best_params_transformed = best_params_transformed
    
    logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"📊 Parámetros EXP: {study.best_params}")
    logger.info(f"🎯 Parámetros TRANSFORMADOS: {best_params_transformed}")
    
    return study

