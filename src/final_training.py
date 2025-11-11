import polars as pl
import lightgbm as lgb
import numpy as np
import logging
from .gain_function import ganancia_evaluator_lgb, ganancia_evaluator_manual
from .config import *


logger = logging.getLogger(__name__)


def evaluar_en_predict(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de predict.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en predict (ganancia + estadísticas básicas)
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
    
    
    #logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    
    # Períodos de evaluación
    periodos_entrenamiento = MES_TRAIN + [202103] + MES_VALIDACION + [202105,202106] #+ MES_TEST
    #periodo_validacion = MES_TEST
    periodo_test = MES_PREDIC
        
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
    
    #X_val = df_val.drop(["clase_ternaria", "clase_01"]).to_pandas()
    #y_val = df_val["clase_01"].to_pandas()
    
    #val_data = lgb.Dataset(X_val, label=y_val) 
    
    # Para test sólo tomo como positivos BAJA+2
    X_test = df_test.drop(["clase_ternaria", "clase_01"]).to_pandas()
    #y_test = df_test["clase_ternaria"].to_pandas()
    #y_test = (y_test == "BAJA+2").astype(int)
    
      
    logger.info(f"Train dataset listo: {X.shape}, Pos: {y.sum()}, Neg: {len(y)-y.sum()} ")
   
    # Listas para almacenar modelos y predicciones
    modelos = []
    predicciones_test = []
    # Entrenar 5 modelos con diferentes semillas
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
        predicciones_test.append(y_pred_proba_single)    

    # Promediar las predicciones de test
    y_pred_proba_ensemble = np.mean(predicciones_test, axis=0)
    
    # Le añado la columna de predicciones al df_test original para análisis posterior si es necesario
    df_test = df_test.with_columns([
        pl.Series("pred_proba_ensemble", y_pred_proba_ensemble)
    ])
    
    # Ordenar por probabilidad de 
    df_test = df_test.sort("pred_proba_ensemble", descending=True)

    # Seleccionar solo las columnas que quieres mantener
    columnas_finales = [
        "numero_de_cliente",                    
        "pred_proba_ensemble"   
    ]
    
 
    return  df_test.select(columnas_finales)