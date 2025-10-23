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
from .gain_function import calcular_ganancia, ganancia_evaluator_lgb,ganancia_evaluator_manual,calcular_ganancia_cortes

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
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + [MES_VALIDACION]
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
    
    periodo_test = MES_TEST
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de Testeo: {periodo_test}")
  
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    df_test = df.filter(pl.col("foto_mes") == periodo_test)
    
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
    

    # ===============================
    # Preparar dataset para LightGBM
    # ===============================
    X = df_sub.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y = df_sub["clase_01"].to_pandas()

    dtrain = lgb.Dataset(X, label=y)
    
    # Para test sólo tomo como positivos BAJA+2
    X_test = df_test.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_test = df_test["clase_ternaria"].to_pandas()
    y_test = (y_test == "BAJA+2").astype(int)
    
    #val_data = lgb.Dataset(X_val, label=y_val)
    
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
            callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)]
            
        )
        
        # Guardar modelo
        modelos.append(model)
        
        # Predecir con este modelo
        y_pred_proba_single = model.predict(X_test)
        predicciones_test.append(y_pred_proba_single)    

    # Promediar las predicciones de test
    y_pred_proba_ensemble = np.mean(predicciones_test, axis=0)  
  
    # Entrenar modelo con mejores parámetros
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()
  
    # Calcular solo la ganancia
    cantidades, ganancias = calcular_ganancia_cortes(y_test, y_pred_proba_ensemble)
  
#    # Estadísticas básicas
#    total_predicciones = len(y_pred_binary)
#    predicciones_positivas = np.sum(y_pred_binary == 1)
#    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
#  
#    resultados = {
#        'ganancia_test': float(ganancia_test),
#        'total_predicciones': int(total_predicciones),
#        'predicciones_positivas': int(predicciones_positivas),
#        'porcentaje_positivas': float(porcentaje_positivas)
#    }
    # Encontrar el índice donde está la ganancia máxima
    indice_max = np.argmax(ganancias)

    #  Obtener la cantidad correspondiente
    cantidad_optima = cantidades[indice_max]
    ganancia_maxima = ganancias[indice_max]


    logger.info(f"La mejor ganancia en test es: {cantidad_optima}, y ocurrio con {ganancia_maxima} envios")
    
#  
    return cantidades, ganancias



def guardar_resultados_test(resultados_test, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo JSON.
    """
    # Guarda en resultados/{STUDY_NAME}_test_results.json
    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana


