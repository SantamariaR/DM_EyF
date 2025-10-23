import numpy as np
import polars as pl
import pandas as pd
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO
import logging

logger = logging.getLogger("__name__")

def ganancia_evaluator_manual(y_true, y_pred):
    """
    Versión para uso manual con arrays numpy/pandas
    """
    # Convertir a arrays numpy
    y_true_np = np.array(y_true).astype(int)
    y_pred_np = np.array(y_pred)
    return calcular_ganancia(y_true_np, y_pred_np)

def calcular_ganancia(y_true, y_pred):
    """
    Función base que calcula la ganancia
    """
    # Crear DataFrame con Polars
    df_eval = pl.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred
    })
  
    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
  
    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col('y_true') == 1)
        .then(GANANCIA_ACIERTO)
        .otherwise(COSTO_ESTIMULO)
        .alias('ganancia_individual')
    ])
  
    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
    ])
  
    # Encontrar la ganancia máxima
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
  
    return ganancia_maxima


def ganancia_evaluator_manual(y_true, y_pred):
    """
    Versión para uso manual con arrays numpy/pandas
    """
    # Convertir a arrays numpy
    y_true_np = np.array(y_true).astype(int)
    y_pred_np = np.array(y_pred)
    return calcular_ganancia(y_true_np, y_pred_np)


def ganancia_evaluator_lgb(preds,dtrain):
    """
    Versión para LightGBM (feval) - usa get_label()
    """
    y_true = dtrain.get_label().astype(int)
    ganancia = calcular_ganancia(y_true, preds)
    return 'ganancia', ganancia, True
