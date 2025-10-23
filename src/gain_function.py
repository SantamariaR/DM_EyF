import numpy as np
import polars as pl
import pandas as pd
from .config import GANANCIA_ACIERTO, COSTO_ESTIMULO, INICIO_ENVIOS, FIN_ENVIOS, PASO_ENVIOS
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

def calcular_ganancia_cortes(y_true, y_pred, start=INICIO_ENVIOS, finish=FIN_ENVIOS, step=PASO_ENVIOS):
    """
    Calcula la ganancia para diferentes cantidades de envíos (cortes)
    
    Args:
        y_true: Array con valores reales (0 o 1)
        y_pred: Array con probabilidades predichas
        start: Cantidad inicial de envíos
        finish: Cantidad final de envíos  
        step: Paso entre cantidades de envíos
    
    Returns:
        tuple: (array_cantidades, array_ganancias)
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
    
    # Generar array de cantidades de envíos
    cantidades = np.arange(start, finish + 1, step)
    
    # Calcular ganancias para cada cantidad de envíos
    ganancias = []
    
    for cantidad in cantidades:
        if cantidad <= len(df_ordenado):
            # Tomar los primeros 'cantidad' registros
            ganancia = df_ordenado.head(cantidad)['ganancia_acumulada'].max()
        else:
            # Si se piden más envíos que registros disponibles, usar el máximo disponible
            ganancia = df_ordenado['ganancia_acumulada'].max()
        
        ganancias.append(ganancia)
    
    return np.array(cantidades), np.array(ganancias)



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
