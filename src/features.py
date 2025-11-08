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
    
    PARAMtrain = {"training": [202101, 202102, 202103]}
    PARAMlgb_param = {
        "num_iterations": 20,
        "num_leaves": 16,
        "min_data_in_leaf": 100,
        "feature_fraction_bynode": 0.2,
        "boosting": "rf",
        "bagging_fraction": (1.0 - 1.0 / np.exp(1.0)),
        "bagging_freq": 1,
        "feature_fraction": 1.0,
        "max_bin": 31,
        "objective": "binary",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
    }

    # Variable clase01
    dataset = dataset.with_columns(
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+3","BAJA+2", "BAJA+1"]))
        .then(1)
        .otherwise(0)
        .alias("clase01")
    )

    campos_buenos = [c for c in dataset.columns if c not in ["clase_ternaria", "clase01"]]

    dataset = dataset.with_columns(
        pl.col("foto_mes").is_in(PARAMtrain["training"]).cast(pl.Int8).alias("entrenamiento")
    )

    train_mask = dataset["entrenamiento"] == 1
    X_train = dataset.filter(train_mask).select(campos_buenos)
    y_train = dataset.filter(train_mask)["clase01"]

    modelo = lgb.train(
        params=PARAMlgb_param,
        train_set=lgb.Dataset(X_train.to_numpy(), label=y_train.to_numpy()),
        callbacks=[lgb.log_evaluation(0)],
    )

    logger.info("Fin construccion RandomForest")

    # === Predicciones para todo el dataset ===
    X_all = dataset.select(campos_buenos).to_numpy()
    pred_leafs = modelo.predict(X_all, pred_leaf=True)  # shape (n_filas, n_arboles)
    n_arboles = pred_leafs.shape[1]

    # === Crear DataFrame con las hojas ===
    df_hojas = pl.DataFrame({
        f"rf_tree_{i:03d}": pred_leafs[:, i] for i in range(n_arboles)
    })

    # === Codificar en variables dummies ===
    df_dummies = df_hojas.to_dummies()

    # === Concatenar con el dataset original ===
    dataset_final = pl.concat([dataset, df_dummies], how="horizontal")

    dataset_final = dataset_final.drop(["clase01"])
    logger.info("Fin AgregaVarRandomForest()")

    return dataset_final
