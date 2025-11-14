#import pandas as pd
import polars as pl
import os
import datetime
import logging

# Funciones personalizadas

from src.loader import cargar_datos,calcular_clase_ternaria,contar_por_grupos,convertir_clase_ternaria_a_target, cargar_features_importantes
from src.features import feature_engineering_lag, feature_engineering_delta_lag,AgregaVarRandomForest,PPR,agregar_suma_m_visa_master
from src.config import *
from src.optimization import optimizar,evaluar_en_test,guardar_resultados_test
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import evaluar_en_predict
from src.output_manager import guardar_resultados_predict
from src.cleaning_features import train_overfit_lgbm_features, add_canaritos, seleccionar_variables_por_canaritos,convertir_todo_cero_a_nan,ajustar_por_inflacion
from src.zfinal_train import evaluamos_en_predict_zlightgbm

# Nombre del log fijo en lugar de uno con timestamp
nombre_log = f"log_{STUDY_NAME}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"{BUCKET_NAME}/log/{nombre_log}", mode="a", encoding="utf-8"),  # Cambiado a "a" (append)
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main2():
    logger.info(f">>> Comienzo del Estudio: {STUDY_NAME} <<<")

    # Asegurar que exista la carpeta de logs
    os.makedirs("logs", exist_ok=True)

    #00 Cargar dataset 
    path_data = DATA_PATH
    df = cargar_datos(path_data)
    logger.info(f"Cargado el dataset:{path_data}")
    
    #Ajuste IPC
    #df = ajustar_por_inflacion(df)
        
    #Tiro algunas columnas que cambien tendencia
    columnas_a_eliminar = ["cprestamos_personales","mprestamos_personales"]
    columnas_base = [col for col in df.columns if col not in columnas_a_eliminar]
    df = df.select(columnas_base)

    # Intento arreglar datadrift
    #df = convertir_todo_cero_a_nan(df)    
   
    #01 Clase ternaria
    df = calcular_clase_ternaria(df)
    logger.info(f"Grupos de clase ternaria por mes:{contar_por_grupos(df)}")
    
        
    #02 Feature Engineering - Lags
    # Ordeno
    df = df.sort(["numero_de_cliente", "foto_mes"])
    # Columnas a excluir
    excluir = ["numero_de_cliente", "foto_mes", "clase_ternaria"]

    # Agrego la suma de los montos
    #df = agregar_suma_m_visa_master(df)
    
    # Obtener columnas para aplicar lags
    columnas_lag = [col for col in df.columns if col not in excluir]
    
    cant_lag = 2
    df = feature_engineering_lag(df, columnas_lag, cant_lag=cant_lag)
    df = feature_engineering_delta_lag(df, columnas_lag, cant_lag=cant_lag)
    
    # Intentamos generar features con PPR
    df = PPR(df)
        
    # Hacemos un RF para agregar variables
    df = AgregaVarRandomForest(df)
    
    logger.info(f"DataFrame final con {len(df.columns)} columnas después de feature engineering")
    
    
   #03 Análisis e features sobre la clase ternaria(la idea es usar canaritos para podar features)
    #logger.info("=== ANÁLISIS DE FEATURES CON CANARITOS ===")
    #df_canaritos,n_canarios = add_canaritos(df,canaritos_ratio=0.5)
    #logger.info(f"Número de canaritos añadidos para análisis: {n_canarios}")
   
    #modelo_canaritos_features = train_overfit_lgbm_features(df_canaritos,undersampling=UNDERSUMPLING)
    #logger.info("Análisis de features con canaritos completado.")
    
    #logger.info(f"DataFrame con canaritos, total columnas: {len(df.columns)}")
    #logger.info(f"Número de canaritos añadidos: {n_canarios}")
    
    # Cargo si es necesario las features importantes según canaritos
    modelo_canaritos_features = cargar_features_importantes(BUCKET_NAME+"/exp/exp48_feature_importance.csv")
    print(modelo_canaritos_features)
    
    logger.info(f"Número de features seleccionadas")    
    df = seleccionar_variables_por_canaritos(modelo_canaritos_features,porcentaje_umbral=0.95,df=df)
    logger.info(f"DataFrame final con {len(df.columns)} columnas después de selección por canaritos")
    
    # Ahora agregamos los canaritos que hace falta para lightgbm
    df,n_canarios = add_canaritos(df,canaritos_ratio=0.05)
    logger.info(f"DataFrame para entrenamiento con zlighgbm:{df.columns}")

    #04 Convertir clase ternaria a target binario
    df = convertir_clase_ternaria_a_target(df)    
    
    # Entrenamiento y evaluación final en modo predict
    df = evaluamos_en_predict_zlightgbm(df,n_canarios=n_canarios)
    
    guardar_resultados_test(df)
    
    
    # Mostrar primeras filas en consola
    print(df.head())

    # Información básica
    filas, columnas = df.shape
    mensaje = f"[{datetime.datetime.now()}] Dataset cargado con {filas} filas y {columnas} columnas\n"
    # Guardar log en archivo
    with open(f"{BUCKET_NAME}/logs_{STUDY_NAME}.txt", "a", encoding="utf-8") as f:
        f.write(mensaje)

    logger.info(f">>> Ejecución finalizada. Revisa logs_{STUDY_NAME}.txt")


if __name__ == "__main__":
    main2()
    
    