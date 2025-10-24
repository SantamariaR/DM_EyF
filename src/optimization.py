import optuna
import lightgbm as lgb
import polars as pl
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_evaluator_lgb,ganancia_evaluator_manual,calcular_ganancia_cortes,calcular_ganancia_acumulada

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
    
      "num_iterations": trial.suggest_int('num_iterations',2048 ,4096 ),
      "learning_rate": trial.suggest_float('learning_rate', 0.002,0.8 ),
      "feature_fraction": trial.suggest_float('feature_fraction',0.2 , 0.8),
      "num_leaves": trial.suggest_int('num_leaves', 2, 82),
      "min_data_in_leaf" : trial.suggest_int('min_data_in_leaf',1 , 2048),  
    }
  
    # Usar target (clase_ternaria ya convertida a binaria)
  
    # Features: usar todas las columnas excepto target
  
    # Entrenar modelo con función de ganancia personalizada
    
    periodos_entrenamiento = MES_TRAIN if isinstance(MES_TRAIN, list) else [MES_TRAIN]
    periodo_validacion = MES_VALIDACION
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de validación: {periodo_validacion}")
        
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))

    # Separar clases
    df_pos = df_train.filter(pl.col("clase_01") == 1)
    df_neg = df_train.filter(pl.col("clase_01") == 0)


    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
#    n_sample = int(df_neg.height * frac)
#    df_neg_sample = df_neg.sample(n=n_sample, seed=SEMILLA + trial.number)

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
    
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
   
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_val = []
    
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
        y_pred_proba_single = model.predict(X_val)
        predicciones_val.append(y_pred_proba_single)

        
        # Calcular ganancia individual
        ganancia_individual = ganancia_evaluator_manual(y_val,y_pred_proba_single)  
        logger.info(f"Ganancia modelo {i+1}: {ganancia_individual}")

    
    
    # Promediar las predicciones de validación
    y_pred_proba_ensemble = np.mean(predicciones_val, axis=0)
    
    logger.info("\n--- Debug de Cositas ---")
    print(f"Predicciones prediocción ensemble (primeros 10): {y_pred_proba_ensemble[:10]}")
    # DEBUG: Verificar el ensemble
    logger.info(f"DEBUG Ensemble - Shape: {y_pred_proba_ensemble.shape}")
    logger.info(f"DEBUG Ensemble - Min: {y_pred_proba_ensemble.min():.4f}, Max: {y_pred_proba_ensemble.max():.4f}")
    logger.info(f"DEBUG Ensemble - y_val shape: {y_val.shape}")
    
    # Calcular ganancia del ensemble
    ganancia_ensemble = ganancia_evaluator_manual(y_val, y_pred_proba_ensemble)
    logger.info(f"\n=== GANANCIA ENSEMBLE: {ganancia_ensemble} ===")    

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_ensemble)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_ensemble:,.0f}")
  
    return ganancia_ensemble


def optimizar(df: pl.DataFrame, n_trials: int = 1) -> optuna.Study:
    """
    Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización ")
    logger.info(f"Configuración: períodos Entrenamiento={MES_TRAIN }, Validación={MES_VALIDACION}")

     # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia(trial, df), n_trials=n_trials)

#    # Normalización de parámetros
#    update_dict = {
#        'min_data_in_leaf': round(study.best_params['min_data_in_leaf'] / HIPERPARAM_BO['UNDERSUMPLING'])
#    }
    #study.best_params.update(update_dict)

    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study



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
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
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
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
#    if isinstance(MES_TRAIN, list):
#        periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
#    else:
#        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
    
    # Períodos de evaliación
    periodos_entrenamiento = MES_TRAIN
    periodo_validacion = MES_VALIDACION
    periodo_test = MES_TEST
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de Validación: {periodo_validacion}")
    logger.info(f"Período de Testeo: {periodo_test}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    df_val = df.filter(pl.col("foto_mes").is_in(periodo_validacion))
    df_test = df.filter(pl.col("foto_mes").is_in(periodo_test))
    
    # Separar clases(por si queremos undersampling )
    df_pos = df_train.filter(pl.col("clase_01") == 1)
    df_neg = df_train.filter(pl.col("clase_01") == 0)


    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
#    n_sample = int(df_neg.height * frac)
#    df_neg_sample = df_neg.sample(n=n_sample, seed=SEMILLA + trial.number)

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
    
    X_val = df_val.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_val = df_val["clase_01"].to_pandas()
    
    val_data = lgb.Dataset(X_val, label=y_val) 
    
    # Para test sólo tomo como positivos BAJA+2
    X_test = df_test.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_test = df_test["clase_ternaria"].to_pandas()
    y_test = (y_test == "BAJA+2").astype(int)
    
      
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
   
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_test = []
    # Entrenar 5 modelos con diferentes semillas
    for i, semilla in enumerate(SEMILLA):
        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
        
        # Actualizar la semilla en los parámetros
        mejores_params['seed'] = semilla
        mejores_params['feature_fraction_seed'] = semilla
        mejores_params['bagging_seed'] = semilla
        
        # Entrenar modelo
        model = lgb.train(
            mejores_params,
            dtrain,
            feval=ganancia_evaluator_lgb,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0),
            lgb.log_evaluation(period=100)]
            
        )
        
        # Guardar modelo
        modelos.append(model)
        
        # Predecir con este modelo
        y_pred_proba_single = model.predict(X_test)
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
    Guarda los resultados de la evaluación en test (DataFrame de curva de ganancia) en un archivo JSON.
    
    Args:
        df_resultado: DataFrame de Polars con la curva de ganancia
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_resultados_test_mes_{MES_TEST}.json"

    # Función para convertir ndarray a lista - VERSIÓN CORREGIDA
    def convertir_a_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
            return [convertir_a_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convertir_a_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convertir DataFrame a diccionario serializable
    datos_dataframe = {}
    for columna in df_resultado.columns:
        datos_dataframe[columna] = convertir_a_serializable(df_resultado[columna].to_list())
    
    # Encontrar el punto de máxima ganancia
    ganancia_maxima = df_resultado['ganancia_acumulada'].max()
    cantidad_optima = df_resultado.filter(
        pl.col('ganancia_acumulada') == ganancia_maxima
    )['cantidad_clientes'].min()
    
    # Crear estructura de datos para guardar
    resultados_test = {
        "study_name": STUDY_NAME,
        "mes_test": MES_TEST,
        "fecha_evaluacion": datetime.now().isoformat(),
        "datos_curva": datos_dataframe,
        "resumen": {
            "ganancia_maxima": float(ganancia_maxima),
            "cantidad_optima_envios": int(cantidad_optima),
            "total_clientes": len(df_resultado),
            "ganancia_final": float(df_resultado['ganancia_acumulada'][-1])
        }
    }

    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r', encoding='utf-8') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(resultados_test)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w', encoding='utf-8') as f:
        json.dump(datos_existentes, f, indent=2, ensure_ascii=False)
  
    logger.info(f"Testeo del MES {MES_TEST} guardado en {archivo}")
    logger.info(f"Ganancia máxima: {ganancia_maxima:,.0f} con {cantidad_optima} envíos")




#def guardar_resultados_test(ganancias,cantidades, archivo_base=None):
#    """
#    Guarda los resultados de la evaluación en test en un archivo JSON.
#    """
#    # Guarda en resultados/{STUDY_NAME}_test_results.json
#    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana
#
#    if archivo_base is None:
#        archivo_base = STUDY_NAME
#  
#    # Nombre del archivo único para todas las iteraciones
#    archivo = f"resultados/{archivo_base}_resultados_test_mes_{MES_TEST}.json"
#
#    # Función para convertir ndarray a lista
#    def convertir_a_serializable(obj):
#        if isinstance(obj, np.ndarray):
#            return obj.tolist()
#        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
#            return int(obj)
#        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
#            return float(obj)
#        elif isinstance(obj, np.bool_):
#            return bool(obj)
#        elif isinstance(obj, dict):
#            return {key: convertir_a_serializable(value) for key, value in obj.items()}
#        elif isinstance(obj, (list, tuple)):
#            return [convertir_a_serializable(item) for item in obj]
#        else:
#            return obj
#    
#    # Convertir los datos a tipos serializables
#    ganancias_serializable = convertir_a_serializable(ganancias)
#    cantidades_serializable = convertir_a_serializable(cantidades)
#    
#    # Crear estructura de datos para guardar
#    resultados_test = {
#        "study_name": STUDY_NAME,
#        "mes_test": MES_TEST,
#        "fecha_evaluacion": datetime.now().isoformat(),
#        "ganancias": ganancias_serializable,
#        "cantidades": cantidades_serializable,
#        "ganancia_total": float(sum(ganancias_serializable)) if ganancias_serializable else 0,
#        "cantidad_total": int(sum(cantidades_serializable)) if cantidades_serializable else 0
#    }
#
#    # Cargar datos existentes si el archivo ya existe
#    if os.path.exists(archivo):
#        with open(archivo, 'r') as f:
#            try:
#                datos_existentes = json.load(f)
#                if not isinstance(datos_existentes, list):
#                    datos_existentes = []
#            except json.JSONDecodeError:
#                datos_existentes = []
#    else:
#        datos_existentes = []
#  
#    # Agregar nueva iteración
#    datos_existentes.append(resultados_test)
#  
#    # Guardar todas las iteraciones en el archivo
#    with open(archivo, 'w') as f:
#        json.dump(datos_existentes, f, indent=2)
#  
#    logger.info(f"Testeo del MES {MES_TEST} guardada en {archivo}")
#    logger.info(f"Ganancia máxima: {np.max(ganancias):,.0f}")
#
