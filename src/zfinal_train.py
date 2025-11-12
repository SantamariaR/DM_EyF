import polars as pl
import lightgbm as lgb
import numpy as np
import logging
from .gain_function import ganancia_evaluator_lgb, ganancia_evaluator_manual
from .config import *
from .gain_function import calcular_ganancia_acumulada


logger = logging.getLogger(__name__)


def evaluamos_en_predict_zlightgbm(df,n_canarios:int) -> dict:
    """
    Evalúa el modelo LightGBM en el conjunto de predict.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        
    Returns:
        dict: Resultados de la evaluación en predict (ganancia + estadísticas básicas) """
        
    
    #Definir hiperparámetros fijos para la evaluación final
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'custom',
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -100,
        
    
        'max_bin': 31,
        'min_data_in_leaf': 20,
        
        'n_estimators': 9999,
        'num_leaves': 999,
        'learning_rate': 1.0,
        
        'feature_fraction': 0.50,
        
        'canaritos': n_canarios,
        'gradient_bound':0.1
    }
        
   
    logger.info("=== EVALUACIÓN EN CONJUNTO DE PREDICCIÓN ===")
    
    # Períodos de evaluación
    periodos_entrenamiento = MES_TRAIN + [202103]+ MES_VALIDACION
    periodo_test = MES_TEST
    frac = UNDERSUMPLING
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
    logger.info(f"Período de Testeo: {periodo_test}")
    logger.info(f"Fracción de undersampling: {frac}")
    
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    df_test = df.filter(pl.col("foto_mes").is_in(periodo_test))
    
    # Separar clases(por si queremos undersampling )
    df_pos = df_train.filter(pl.col("clase_01") == 1)
    df_neg = df_train.filter(pl.col("clase_01") == 0)


    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
    n_sample = int(df_neg.height * frac)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

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
    
    # Para test sólo tomo como positivos BAJA+2
    X_test = df_test.drop(["clase_ternaria", "clase_01"]).to_pandas()
    y_test = df_test["clase_ternaria"].to_pandas()
    y_test = (y_test == "BAJA+2").astype(int)
    
      
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
    
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_test = []
    # Entrenar 5 modelos con diferentes semillas
    for i, semilla in enumerate(SEMILLA[:2]):#+SEMILLERO):
        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
        
        # Actualizar la semilla en los parámetros
        params['seed'] = semilla
        params['feature_fraction_seed'] = semilla
        params['bagging_seed'] = semilla
        
        model = lgb.train(
            params,
            dtrain,
            feval=ganancia_evaluator_lgb,
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        # Guardar modelo
        modelos.append(model)
        
        # ✅ CORRECCIÓN: Réplica exacta del comportamiento de R
        # En R: predict(modelo_final, data.matrix(dfuture[, campos_buenos, with= FALSE]))
        #X_test_matrix = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Opción 1: Probabilidades (como en R)
        y_pred_proba_single =  abs(model.predict(X_test))
        
        # Debugging de las predicciones
        logger.info(f"Rango predicciones modelo {i+1}: [{y_pred_proba_single.min():.4f}, {y_pred_proba_single.max():.4f}]")
        logger.info(f"Predicciones modelo {i+1} (primeros 5): {[f'{x:.4f}' for x in y_pred_proba_single[:5]]}")
        
#        # Verificar si hay valores fuera de [0,1]
#        if y_pred_proba_single.min() < 0 or y_pred_proba_single.max() > 1:
#            logger.warning(f"⚠️ Modelo {i+1} tiene predicciones fuera de [0,1]")
#            # Aplicar sigmoid si es necesario (pero raw_score=False debería evitarlo)
#            y_pred_proba_single = 1 / (1 + np.exp(-y_pred_proba_single))
        
        predicciones_test.append(y_pred_proba_single)

    # Promediar las predicciones
    y_pred_proba_ensemble = np.mean(predicciones_test, axis=0)
    
    # ✅ VERIFICAR el ensemble también
    logger.info(f"Rango predicciones ensemble: [{y_pred_proba_ensemble.min():.4f}, {y_pred_proba_ensemble.max():.4f}]")
    logger.info(f"Predicciones ensemble (primeras 10): {[f'{x:.4f}' for x in y_pred_proba_ensemble[:10]]}")
    
    # Verificar distribución
    unique_values = np.unique(y_pred_proba_ensemble.round(4))
    logger.info(f"Valores únicos en predicciones (redondeados a 4 decimales): {unique_values}")

    
    # Le añado la columna de predicciones al df_test original para análisis posterior si es necesario
    df_test = df_test.with_columns([
        pl.Series("pred_proba_ensemble", y_pred_proba_ensemble)
    ])
    
    #logger.info(f"Datos de test con predicciones listos: {df_test.head()}")
    # Calcular solo la ganancia
    
    # Calculamos las ganancias
    df_pred = calcular_ganancia_acumulada(df_test, col_probabilidad="pred_proba_ensemble", col_clase="clase_ternaria")
 
    return  df_pred