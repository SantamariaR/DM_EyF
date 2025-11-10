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

#def calcular_ganancia(y_true, y_pred):
#    """
#    Función base que calcula la ganancia
#    """
#    # Crear DataFrame con Polars
#    df_eval = pl.DataFrame({
#        'y_true': y_true,
#        'y_pred_proba': y_pred
#    })
#  
#    # Ordenar por probabilidad descendente
#    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
#  
#    # Calcular ganancia individual para cada cliente
#    df_ordenado = df_ordenado.with_columns([
#        pl.when(pl.col('y_true') == 1)
#        .then(GANANCIA_ACIERTO)
#        .otherwise(COSTO_ESTIMULO)
#        .alias('ganancia_individual')
#    ])
#  
#    # Calcular ganancia acumulada
#    df_ordenado = df_ordenado.with_columns([
#        pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
#    ])
#  
#    # Encontrar la ganancia máxima
#    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
#  
#    return ganancia_maxima

def calcular_ganancia(y_true, y_pred, ventana=1000):
    """
    Función que calcula el promedio de ganancias en una ventana alrededor del máximo
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
  
    # Encontrar la posición del máximo
    indice_maximo = df_ordenado.select(
        pl.col('ganancia_acumulada').arg_max()
    ).item()
    
    # Calcular los límites de la ventana
    inicio_ventana = max(0, indice_maximo - ventana)
    fin_ventana = min(len(df_ordenado) - 1, indice_maximo + ventana)
    
    # Obtener las ganancias en la ventana
    ganancias_ventana = df_ordenado.slice(inicio_ventana, fin_ventana - inicio_ventana + 1)\
        .select(pl.col('ganancia_acumulada'))\
        .to_series()
    
    # Calcular el promedio
    ganancia_promedio = ganancias_ventana.mean()
    
    return ganancia_promedio


def calcular_ganancia_acumulada(df, col_probabilidad="pred_proba_ensemble", col_clase="clase_ternaria"):
    """
    Calcula la ganancia acumulada ordenando por probabilidad (mayor a menor)
    y sumando/restando según la clase.
    
    Args:
        df: DataFrame de Polars
        col_probabilidad: Columna con las probabilidades del ensemble
        col_clase: Columna con las clases (0 o 1)
    
    Returns:
        DataFrame con la ganancia acumulada
    """
    
    # Ordenar por probabilidad de 
    df_ordenado = df.sort(col_probabilidad, descending=True)
    
    # Calcular ganancia individual
    df_con_ganancia = df_ordenado.with_columns([
        pl.when(pl.col(col_clase) == "BAJA+2")
        .then(GANANCIA_ACIERTO)  # Sumar si es BAJA+2
        .otherwise(COSTO_ESTIMULO)  # Restar si es 0
        .alias("ganancia_individual")
    ])
    
    # Calcular ganancia acumulada
    df_resultado = df_con_ganancia.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada"),
        (pl.arange(0, pl.len()) + 1).alias("cantidad_clientes")
    ])
    
    # Seleccionar solo las columnas que quieres mantener
    columnas_finales = [
        "numero_de_cliente",      
        col_clase,                  
        col_probabilidad,         
        "ganancia_individual",    
        "ganancia_acumulada",     
        "cantidad_clientes"       
    ]
    
    return df_resultado.select(columnas_finales)



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
