import polars as pl
import lightgbm as lgb
import numpy as np
import logging
from .gain_function import ganancia_evaluator_lgb, ganancia_evaluator_manual
from .config import *
from .gain_function import calcular_ganancia_acumulada,calcular_ganancia



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
    periodos_entrenamiento = MES_TRAIN + [202103] + MES_VALIDACION
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
    for i, semilla in enumerate(SEMILLA): # +SEMILLERO):
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
        gan_y_pred = calcular_ganancia(y_test,y_pred_proba_single)
        
        # Debugging de las predicciones
        logger.info(f"Ganancia promedio de la cresta {gan_y_pred}")
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


def evaluamos_en_final_zlightgbm(df,n_canarios:int) -> dict:
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
    periodos_entrenamiento = MES_TRAIN + [202103] + MES_VALIDACION + [202105] + MES_TEST
    periodo_test = MES_PREDIC
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
    for i, semilla in enumerate(SEMILLA): # +SEMILLERO):
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
        gan_y_pred = calcular_ganancia(y_test,y_pred_proba_single)
        
        # Debugging de las predicciones
        logger.info(f"Ganancia promedio de la cresta {gan_y_pred}")
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








def evaluamos_en_predict_zlightgbm_amputado(df,n_canarios:int) -> dict:
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
        #'first_metric_only': False,
        #'boost_from_average': True,
        #'feature_pre_filter': False,
        #'force_row_wise': True,
        #'verbosity': -100,
        
    
        'max_bin': 31,
        'min_data_in_leaf': 200,
        
        'n_estimators': 64,
        #'num_leaves': 999,
        'learning_rate': 1.0,
        
        #'feature_fraction': 0.50,
        
        'canaritos': n_canarios,
        'gradient_bound':0.4
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
    for i, semilla in enumerate(SEMILLERO): # +SEMILLERO):
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
        gan_y_pred = calcular_ganancia(y_test,y_pred_proba_single)
        
        # Debugging de las predicciones
        logger.info(f"Ganancia promedio de la cresta {gan_y_pred}")
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


def generar_proba_rolling_lightgbm(
    df: pl.DataFrame,
    n_canarios: int = 5,
    meses_contexto: int = 4
) -> pl.DataFrame:

    logger.info("=== INICIANDO ROLLING LIGHTGBM (con ensemble y canarios) ===")

    # Ordeno meses disponibles
    meses = MES_TRAIN + [202103]+ MES_VALIDACION +[202105] + MES_TEST

    df = df.with_row_index("row_id")   # versión nueva y no-deprecada
    df_out = df.clone().with_columns(
        pl.lit(None).alias("pred_proba_rolling")
    )

    predicciones = {}  # <-- acá se inicializa

    # Features para LightGBM
    features = [
        c for c in df.columns
        if c not in ["foto_mes", "clase_ternaria", "clase_01"]
    ]

    # LightGBM base params
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
        'gradient_bound': 0.1
    }

    # Loop por cada mes
    for mes_actual in meses:
        
        meses_train = [
            m for m in meses
            if (m < mes_actual) and (m >= mes_actual - meses_contexto)
        ]

        if len(meses_train) == 0:
            logger.info(f"Mes {mes_actual}: sin contexto → NaN")
            continue

        logger.info(f"\n--- Mes {mes_actual} ---")
        logger.info(f"Meses usados como train: {meses_train}")

        # Train
        df_train = df_out.filter(pl.col("foto_mes").is_in(meses_train))
        df_mes   = df_out.filter(pl.col("foto_mes") == mes_actual)

        if df_train.is_empty() or df_mes.is_empty():
            logger.warning(f"Mes {mes_actual}: train o test vacío → skip")
            continue

        # Undersampling
        df_pos = df_train.filter(pl.col("clase_01") == 1)
        df_neg = df_train.filter(pl.col("clase_01") == 0)

        n_sample = int(df_neg.height * UNDERSUMPLING)
        df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

        df_sub = (
            pl.concat([df_pos, df_neg])
            .sample(fraction=1.0, shuffle=True, seed=SEMILLA[0])
        )

        # Pandas para LGBM
        X = df_sub[features].to_pandas()
        y = df_sub["clase_01"].to_pandas()
        X_mes = df_mes[features].to_pandas()

        logger.info(f"Entrenando con {X.shape[0]} filas...")

        # Ensemble
        pred_list = []

        for semilla in SEMILLA:
            logger.info(f"  Modelo seed={semilla}")

            params["seed"] = semilla
            params["feature_fraction_seed"] = semilla
            params["bagging_seed"] = semilla

            dtrain = lgb.Dataset(X, label=y)

            model = lgb.train(
                params,
                dtrain,
                feval=ganancia_evaluator_lgb,
                callbacks=[lgb.log_evaluation(period=100)]
            )

            pred = abs(model.predict(X_mes))
            pred_list.append(pred)

            logger.info(f"pred: min={pred.min():.4f}, max={pred.max():.4f}")

        # Ensemble final
        pred_ensemble = np.mean(pred_list, axis=0)
        logger.info(f"Pred ensemble mes {mes_actual}: min={pred_ensemble.min():.4f}, max={pred_ensemble.max():.4f}")

        row_ids = df_mes["row_id"].to_list()
        preds_list = np.asarray(pred_ensemble, dtype=np.float32).tolist()
        predicciones[mes_actual] = (row_ids, preds_list)
        
           
    df_pred_list = []
    for mes, (row_ids, preds) in predicciones.items():
        df_pred_list.append(
            pl.DataFrame({
                "row_id": row_ids,
                "pred_proba_rolling": preds
            })
        )

    if df_pred_list:
        df_pred = pl.concat(df_pred_list)
    else:
        df_pred = pl.DataFrame({"row_id": [], "pred_proba_rolling": []})

    # Unir UNA vez por row_id (mucho más eficiente)
    df_out = df_out.join(df_pred, on="row_id", how="left").with_columns(
        pl.when(pl.col("pred_proba_rolling").is_not_null())
        .then(pl.col("pred_proba_rolling"))
        .otherwise(pl.col("pred_proba_rolling"))  # ya está en la columna; esto es sólo para ejemplificar
    )

    # si ya no necesitás row_id, podés dropearla
    df_out = df_out.drop("row_id")
    
    logger.info("=== ROLLING LGBM FINALIZADO ===")
    return df_out
