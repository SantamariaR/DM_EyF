#import pandas as pd
import polars as pl
import os
import datetime
import logging

# Funciones personalizadas

from src.loader import cargar_datos,calcular_clase_ternaria,contar_por_grupos,convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag, feature_engineering_delta_lag
from src.config import *
from src.optimization import optimizar


## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    print(">>> Inicio de ejecución")

    # Asegurar que exista la carpeta de logs
    os.makedirs("logs", exist_ok=True)

    #00 Cargar dataset 
    path_data = DATA_PATH    
    df = cargar_datos(path_data)
    logger.info(f"Cargado el dataset:{path_data}")
    
    #01 Clase ternaria
    df = calcular_clase_ternaria(df)
    logger.info(f"Grupos de clase ternaria por mes:{contar_por_grupos(df)}")
    
        
    #02 Feature Engineering - Lags
    columnas_lag = ["ctrx_quarter"]
    cant_lag = 1
    df = feature_engineering_lag(df, columnas_lag, cant_lag=cant_lag)
    df = feature_engineering_delta_lag(df, columnas_lag, cant_lag=cant_lag)
    
    #03 Convertir clase ternaria a target binario
    df = convertir_clase_ternaria_a_target(df)
    
    # 4. Ejecutar optimización (función simple)
    study = optimizar(df, n_trials= 50) 
    
    # 5. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")    
    #02 Guardar dataset procesado
    #path_salida = "data/competencia_01_procesado.csv"
    #df.write_csv(path_salida)

    # Mostrar primeras filas en consola
    print(df.head())

    # Información básica
    filas, columnas = df.shape
    mensaje = f"[{datetime.datetime.now()}] Dataset cargado con {filas} filas y {columnas} columnas\n"

    # Guardar log en archivo
    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(mensaje)

    print(">>> Ejecución finalizada. Revisa logs/logs.txt")


if __name__ == "__main__":
    main()
