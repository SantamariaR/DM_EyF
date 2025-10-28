import pandas as pd
import polars as pl
import os
import logging
from datetime import datetime
from .config import *

logger = logging.getLogger(__name__)

def guardar_resultados_predict(df_resultado, archivo_base=None):
    """
    Guarda los resultados de la evaluación en test en un archivo CSV.
    
    Args:
        df_resultado: DataFrame de Polars con la curva de ganancia
        archivo_base: Nombre base del archivo (si es None, usa STUDY_NAME)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
 
     # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "exp")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    archivo_csv = os.path.join(path_db, f"{archivo_base}_predict_mes_{MES_PREDIC}.csv")

    
    # Crear directorio si no existe
    os.makedirs("predict", exist_ok=True)
    
    # Guardar directamente como CSV desde Polars
    df_resultado.write_csv(archivo_csv)
    
    logger.info(f"Testeo del MES {MES_PREDIC} guardado en {archivo_csv}")
    logger.info(f"Total de registros: {len(df_resultado):,}")