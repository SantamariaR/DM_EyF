#import pandas as pd
import polars as pl
import os
import datetime
import logging

# Funciones personalizadas

from src.loader import cargar_datos,calcular_clase_ternaria,contar_por_grupos,convertir_clase_ternaria_a_target, cargar_features_importantes
from src.features import feature_engineering_lag, feature_engineering_delta_lag
from src.config import *
from src.optimization import optimizar,evaluar_en_test,guardar_resultados_test
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import evaluar_en_predict
from src.output_manager import guardar_resultados_predict
from src.cleaning_features import train_overfit_lgbm_features, add_canaritos, seleccionar_variables_por_canaritos,estandarizar_variables_monetarias_polars,convertir_ceros_a_nan


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

def main():
    print(">>> Inicio de ejecución")

    # Asegurar que exista la carpeta de logs
    os.makedirs("logs", exist_ok=True)

    #00 Cargar dataset 
    path_data = DATA_PATH
    df = cargar_datos(path_data)
    logger.info(f"Cargado el dataset:{path_data}")
    
    # Intento arreglar datadrift
#    df = estandarizar_variables_monetarias_polars(df)
    df = convertir_ceros_a_nan(df, columna_mes='foto_mes', umbral_ceros=0.99)
#   Tiro algunas columnas que cambien tendencia
    columnas_a_eliminar = ["cprestamos_personales","mprestamos_personales"]
    columnas_base = [col for col in df.columns if col not in columnas_a_eliminar]
    df = df.select(columnas_base)

   
    #01 Clase ternaria
    df = calcular_clase_ternaria(df)
    logger.info(f"Grupos de clase ternaria por mes:{contar_por_grupos(df)}")
    
        
    #02 Feature Engineering - Lags
    # Ordeno y después genero lags
    df = df.sort(["numero_de_cliente", "foto_mes"])
    # Columnas a excluir
    excluir = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    # Obtener columnas para aplicar lags
    columnas_lag = [col for col in df.columns if col not in excluir]
    
#    columnas_lag = ["ctrx_quarter"]
    cant_lag = 2
    df = feature_engineering_lag(df, columnas_lag, cant_lag=cant_lag)
    df = feature_engineering_delta_lag(df, columnas_lag, cant_lag=cant_lag)
    
   #03 Análisis e features sobre la clase ternaria(la idea es usar canaritos para podar features)
    #df_canaritos = add_canaritos(df,canaritos_ratio=0.5)
   
    #modelo_canaritos_features = train_overfit_lgbm_features(df_canaritos,undersampling=UNDERSUMPLING)
    
    # Cargo si es necesario las features importantes según canaritos
    modelo_canaritos_features = cargar_features_importantes(BUCKET_NAME+"/exp/exp22_feature_importance.csv")
    #print(modelo_canaritos_features)
    
    df = seleccionar_variables_por_canaritos(modelo_canaritos_features,porcentaje_umbral=0.5,df=df)
 
   
    #04 Convertir clase ternaria a target binario
    df = convertir_clase_ternaria_a_target(df)
    
    #05 . Ejecutar optimización (función simple)
    study = optimizar(df, n_trials= 30,undersampling=UNDERSUMPLING) 
    
    #06. Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    #07 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    #mejores_params = cargar_mejores_hiperparametros()
    # Acceder a los parámetros transformados
    mejores_params = study.best_params_transformed
    mejor_ganancia = study.best_value
    
    print(f"Mejor ganancia: {mejor_ganancia:,.0f}")
    print(f"Parámetros para el modelo: {mejores_params}")
  
    # Evaluar en test
    df_test = evaluar_en_test(df, mejores_params)
  
    # Guardar resultados de test
    guardar_resultados_test(df_test)
    
    logger.info("=== EVALUACIÓN EN TEST COMPLETADA ===")
    logger.info("=== COMIENZA PREDICCION FINAL PARA ENVÍOS ===")

    #06 Predicción para envíos    
    df_predic = evaluar_en_predict(df, mejores_params)
    
    guardar_resultados_predict(df_predic)
    
    
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
    main()
