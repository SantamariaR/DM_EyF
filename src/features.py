#import pandas as pd
#import duckdb
import logging
import polars as pl
import numpy as np
import lightgbm as lgb

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando Polars nativo.
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas)} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    # Filtrar columnas que existen en el DataFrame
    columnas_existentes = [col for col in columnas if col in df.columns]
    
    if not columnas_existentes:
        logger.warning("Ninguno de los atributos especificados existe en el DataFrame")
        return df

    # Generar expresiones de lag
    lag_expressions = []
    for attr in columnas_existentes:
        for i in range(1, cant_lag + 1):
            lag_expr = pl.col(attr).shift(i).over("numero_de_cliente").alias(f"{attr}_lag_{i}")
            lag_expressions.append(lag_expr)

    # Aplicar los lags en una sola operación
    df_result = df.with_columns(lag_expressions)

    logger.info(f"LAGS completado. DataFrame resultante con {len(df_result.columns)} columnas")
    
    return df_result



def feature_engineering_delta_lag(df: pl.DataFrame, columnas: list[str], cant_lag: int = 1) -> pl.DataFrame:
    """
    Versión optimizada que calcula delta lags directamente sin columnas intermedias.
    """
    if not columnas:
        return df

    # Filtrar solo columnas numéricas existentes
    columnas_numericas = []
    for col in columnas:
        if col in df.columns:
            if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                pl.Float32, pl.Float64]:
                columnas_numericas.append(col)
            else:
                logger.warning(f"Columna {col} ignorada porque no es numérica (tipo: {df[col].dtype})")
    
    if not columnas_numericas:
        return df

    delta_expressions = []
    for attr in columnas_numericas:
        for i in range(1, cant_lag + 1):
            current_lag = pl.col(attr).shift(i-1).over("numero_de_cliente")
            next_lag = pl.col(attr).shift(i).over("numero_de_cliente")
            delta_expr = (current_lag - next_lag).alias(f"{attr}_delta_lag_{i}")
            delta_expressions.append(delta_expr)

    df_result = df.with_columns(delta_expressions)
    
    logger.info(f"Delta LAGS completado. DataFrame resultante con {len(df_result.columns)} columnas")
    logger.info(f"Columnas procesadas: {columnas_numericas}")

    return df_result


def AgregaVarRandomForest(dataset: pl.DataFrame) -> pl.DataFrame:
    logger.info("inicio AgregaVarRandomForest()")
    
    # Parámetros (debes definirlos antes de llamar a la función)
    PARAMtrain = {"training": [202101, 202102, 202103]}
    PARAMarbolitos = 20
    PARAMhojas_por_arbol = 16
    PARAMdatos_por_hoja = 100
    PARAMmtry_ratio = 0.2
    
    PARAMlgb_param = {
        # parametros que se pueden cambiar
        "num_iterations": PARAMarbolitos,
        "num_leaves": PARAMhojas_por_arbol,
        "min_data_in_leaf": PARAMdatos_por_hoja,
        "feature_fraction_bynode": PARAMmtry_ratio,
        
        # para que LightGBM emule Random Forest
        "boosting": "rf",
        "bagging_fraction": (1.0 - 1.0 / np.exp(1.0)),
        "bagging_freq": 1,
        "feature_fraction": 1.0,
        
        # genericos de LightGBM
        "max_bin": 31,
        "objective": "binary",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        
        "pos_bagging_fraction": 1.0,
        "neg_bagging_fraction": 1.0,
        "is_unbalance": False,
        "scale_pos_weight": 1.0,
        
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        
        "extra_trees": False
    }
    
    # Crear variable clase01
    dataset = dataset.with_columns(
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]))
        .then(1)
        .otherwise(0)
        .alias("clase01")
    )
    
    # Definir campos buenos (excluyendo clase_ternaria y clase01)
    campos_buenos = [col for col in dataset.columns if col not in ["clase_ternaria", "clase01"]]
    
    # Crear variable entrenamiento
    dataset = dataset.with_columns(
        pl.col("foto_mes").is_in(PARAMtrain["training"]).cast(pl.Int8).alias("entrenamiento")
    )
    
    # Preparar datos de entrenamiento
    train_mask = dataset["entrenamiento"] == 1
    X_train = dataset.filter(train_mask).select(campos_buenos)
    y_train = dataset.filter(train_mask)["clase01"]
    
    # Convertir a numpy para LightGBM
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    
    # Crear Dataset de LightGBM
    dtrain = lgb.Dataset(
        data=X_train_np,
        label=y_train_np,
        free_raw_data=False
    )
    
    # Entrenar modelo
    modelo = lgb.train(
        params=PARAMlgb_param,
        train_set=dtrain,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    logger.info("Fin construccion RandomForest")
    
    # Guardar modelo
    #modelo.save_model("modelo.model")
    
    qarbolitos = PARAMlgb_param["num_iterations"]
    periodos = dataset["foto_mes"].unique().to_list()
    
    # Hacer predicciones por periodo
    for periodo in periodos:
        logger.info(f"periodo = {periodo}")
        
        # Filtrar datos del periodo
        periodo_mask = dataset["foto_mes"] == periodo
        X_periodo = dataset.filter(periodo_mask).select(campos_buenos)
        X_periodo_np = X_periodo.to_numpy()
        
        print("Inicio prediccion")
        # Predecir hojas (predicción tipo "leaf")
        prediccion = modelo.predict(
            X_periodo_np,
            pred_leaf=True
        )
        logger.info("Fin prediccion")
        
        # Para cada árbol
        for arbolito in range(qarbolitos):
            logger.info(f"{arbolito + 1} ", end="")
            
            # Obtener hojas únicas para este árbol
            hojas_arbol = np.unique(prediccion[:, arbolito])
            
            # Para cada hoja en el árbol
            for pos, nodo_id in enumerate(hojas_arbol):
                # Crear nombre de variable
                var_name = f"rf_{arbolito + 1:03d}_{nodo_id:03d}"
                
                # Crear máscara para esta hoja
                mask_hoja = (prediccion[:, arbolito] == nodo_id).astype(int)
                
                # Agregar columna al dataset
                # Necesitamos mapear de vuelta a los índices originales del periodo
                periodo_indices = dataset.filter(periodo_mask).select(pl.first()).to_series().to_list()
                
                # Crear serie completa con ceros y unos en las posiciones correctas
                full_series = pl.Series([0] * len(dataset))
                for i, idx in enumerate(periodo_indices):
                    if i < len(mask_hoja) and mask_hoja[i] == 1:
                        full_series = full_series.set(idx, 1)
                
                dataset = dataset.with_columns(full_series.alias(var_name))
            
        print("\n")
    
    # Eliminar columna clase01
    dataset = dataset.drop("clase01")
    
    print("Fin AgregaVarRandomForest()")
    return dataset
