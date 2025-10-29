#import pandas as pd
#import duckdb
import logging
import polars as pl

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