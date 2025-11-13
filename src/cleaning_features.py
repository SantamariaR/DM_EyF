# src/cleaning_features.py
# Módulo de funciones para la limpieza de features con canarios y otras técnicas. Usando Polars.
import lightgbm as lgb
import numpy as np
import logging
import polars as pl
from typing import List
from .config import MES_TRAIN, MES_VALIDACION, SEMILLA, BUCKET_NAME, STUDY_NAME
from .gain_function import ganancia_evaluator_lgb
import os
import pandas as pd
import scipy.interpolate as interp

logger = logging.getLogger(__name__)

def add_canaritos(df: pl.DataFrame, 
                  seed: int = 102191,
                  canaritos_ratio: float = 0.5,
                  columns_to_ignore: List[str] = None) -> tuple[pl.DataFrame, int]:
    """
    Réplica exacta del comportamiento de R que muestras:
    - Elimina columnas específicas
    - Añade columna 'azar' con valores uniformes
    - Ordena el DataFrame por 'azar' (manteniendo relaciones entre variables)
    - Elimina 'azar'
    - Añade features canaritos y las coloca al principio
    
    Args:
        df: DataFrame original de Polars
        seed: Semilla para reproducibilidad (102191 por defecto)
        canaritos_ratio: Proporción de features canaritos
        columns_to_ignore: Columnas a ignorar antes de añadir canaritos
    
    Returns:
        DataFrame con canaritos al principio y orden aleatorio
    """
    # Configurar semilla
    np.random.seed(seed)
    
    # Columnas por defecto a ignorar
    if columns_to_ignore is None:
        columns_to_ignore = ['numero_de_cliente', 'clase_ternaria', 'foto_mes']
    
    # 1. Crear copia y añadir columna 'azar'
    azar_values = np.random.uniform(0, 1, len(df))
    df_canaritos = df.with_columns(
        pl.Series('azar', azar_values)
    )
    
    # 2. Ordenar por 'azar' (mantiene relaciones entre variables)
    df_canaritos = df_canaritos.sort('azar')
    
    # 3. Calcular número de canaritos basado en columnas NO ignoradas
    # Excluimos 'azar' y las columnas a ignorar del cálculo
    columns_for_canaritos_count = [col for col in df.columns 
                                   if col not in columns_to_ignore]
    n_original_features = len(columns_for_canaritos_count)
    n_canaritos = max(1, int(n_original_features * canaritos_ratio))
    
    # 4. Añadir features canaritos al DataFrame ordenado
    canaritos_names = []
    for i in range(n_canaritos):
        random_data = np.random.normal(0, 1, len(df_canaritos))
        canarito_name = f'canarito_{i:03d}'
        canaritos_names.append(canarito_name)
        
        df_canaritos = df_canaritos.with_columns(
            pl.Series(canarito_name, random_data)
        )
    
    # 5. Eliminar columna 'azar' (ya cumplió su función de ordenar)
    df_canaritos = df_canaritos.drop('azar')
    
    # 6. REORDENAR COLUMNAS: Canaritos primero, luego el resto
    all_columns = df_canaritos.columns
    # Separar canaritos del resto de columnas
    non_canaritos_columns = [col for col in all_columns if col not in canaritos_names]
    # Crear nuevo orden: canaritos primero, luego las demás columnas
    new_column_order = canaritos_names + non_canaritos_columns
    # Reordenar el DataFrame
    df_canaritos = df_canaritos.select(new_column_order)
    
    print(f"✅ Proceso completado con semilla: {seed}")
    print(f"📊 Columnas ignoradas: {columns_to_ignore}")
    print(f"🎯 Features originales (para cálculo): {n_original_features}")
    print(f"🔔 Features canaritos añadidas: {n_canaritos}")
    print(f"📈 Features totales: {len(df_canaritos.columns)}")
    print(f"📍 Canaritos colocados al principio del dataset")
    
    return df_canaritos, n_canaritos



def train_overfit_lgbm_features(df: pl.DataFrame, objective: str = 'binary', undersampling: float = 0.5,archivo_base=None) -> dict:
    
    """
    Entrenamos un modelo lightGBM hasta el sobreajuste total para detectar features inútiles (canaritos).
    Args:
        df_train: DataFrame de Polars con datos de entrenamiento (incluye target)
        objective: Objetivo de LightGBM (por defecto 'multiclass')
        num_class: Número de clases (por defecto 3)
    Returns:
        Modelo LightGBM entrenado hasta sobreajuste
    """
    params = {
        'objective': objective,
        #'metric': 'binary_logloss',
        #'num_class': num_class,
        'boosting_type': 'gbdt',
        
       
        'num_leaves': 2**10,          
        'max_depth': -1,               
        'min_data_in_leaf': 1,       
        'min_child_samples': 1,      
        'min_split_gain': 0.0,         
        
    
        'lambda_l1': 0.0,             
        'lambda_l2': 0.0,             
        

        'feature_fraction': 1.0,       
        'bagging_fraction': 1.0,    
        'bagging_freq': 0,             
        'learning_rate': 0.1,
        'verbosity': -100
    }

    
    # Períodos de evaluación
    #mitad = (len(MES_TRAIN) + 1) // 2
    #periodos_entrenamiento = MES_TRAIN[:mitad]
    periodos_entrenamiento = [202008,202009, 202010, 202011,202012]
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
 
    # Data preparación, train y test
    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
        
    # Separar clases
#    df_pos = df_train.filter(pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"]))
#    df_neg = df_train.filter(pl.col("clase_ternaria") == "CONTINUA")
    
    # Separar clases
    df_pos = df_train.filter(pl.col("clase_ternaria").is_in(["BAJA+2","BAJA+1"]))
    df_neg = df_train.filter(pl.col("clase_ternaria").is_in(["CONTINUA"]))

    # Polars no tiene sample(frac=...), pero podemos calcular cuántas filas queremos
    n_sample = int(df_neg.height * undersampling)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_sub = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0])
    # ==================================================
    # Preparar dataset para LightGBM, entrenar y testear
    # ==================================================
    # Mapeo clase_ternaria a numérico
    mapping = {'CONTINUA': 0, 'BAJA+1': 1, 'BAJA+2': 1}
    
    X = df_sub.drop(["clase_ternaria","numero_de_cliente"]).to_pandas()
    y = df_sub["clase_ternaria"].to_pandas().map(mapping)

    dtrain = lgb.Dataset(X, label=y)
    
    # Diccionario para acumular importancias
    feature_importance_total = {}
    
    #modelos = []
    
    # Entrenar 5 modelos con diferentes semillas
    for i, semilla in enumerate(SEMILLA):
        logger.info(f"\n--- Entrenando modelo {i+1} con semilla {semilla} ---")
        
        # Actualizar la semilla en los parámetros
        params['seed'] = semilla
        params['feature_fraction_seed'] = semilla
        params['bagging_seed'] = semilla
        
        # Entrenar modelo
        model = lgb.train(
            params,
            dtrain,
            feval=ganancia_evaluator_lgb,
            num_boost_round=100            
        )
        #modelos.append(model)
        # Obtener importancia de features
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        # Acumular importancias
        for feature, imp in zip(feature_names, importance):
            if feature not in feature_importance_total:
                feature_importance_total[feature] = 0
            feature_importance_total[feature] += imp
        
        logger.info(f"✅ Modelo {i+1}/{5} completado (seed: {semilla})")
        
    # Calcular importancia promedio
    feature_importance_avg = {k: v / 5 for k, v in feature_importance_total.items()}
    
    # Ordenar por importancia
    feature_importance_sorted = dict(sorted(
        feature_importance_avg.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    logger.info(f"\n📊 Resumen de importancia de features ({5} modelos):")
    logger.info(f"🔝 Top 10 features más importantes:")
    for i, (feature, importance) in enumerate(list(feature_importance_sorted.items())[:10]):
        logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    logger.info(f"\n🔚 Bottom 10 features menos importantes:")
    for i, (feature, importance) in enumerate(list(feature_importance_sorted.items())[-10:]):
        logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "exp")
    os.makedirs(path_db, exist_ok=True)
    
    # Ruta completa de la base de datos
    archivo_csv = os.path.join(path_db, f"{archivo_base}_feature_importance.csv")
    
    # ✅ FORMA CORRECTA: Crear DataFrame con dos columnas
    df_feature_importance = pd.DataFrame(
        list(feature_importance_sorted.items()),  # Convertir a lista de tuplas
        columns=['Feature', 'Importance']         # Nombres de columnas
    )
    
    # ✅ FORMA ALTERNATIVA: Desde el diccionario original
    # df_feature_importance = pd.DataFrame([
    #     {'Feature': k, 'Importance': v} 
    #     for k, v in feature_importance_sorted.items()
    # ])
    
    df_feature_importance.to_csv(archivo_csv, index=False)
    logger.info(f"✅ Feature importance guardado en: {archivo_csv}")
    return feature_importance_sorted



def seleccionar_variables_por_canaritos(feature_importance_sorted: dict, 
                                      df: pl.DataFrame, 
                                      porcentaje_umbral: float = 0.05) -> pl.DataFrame:
    """
    Versión optimizada que evita duplicados usando conjuntos
    """
    # Identificar canaritos
    canaritos = [feature for feature in feature_importance_sorted.keys() 
                if feature.startswith('canarito')]
    
    if not canaritos:
        logger.warning("⚠️ No se encontraron variables canarito en las importancias")
        return df
    
    # Calcular umbral
    importancias_canaritos = [feature_importance_sorted[canarito] for canarito in canaritos]
    percentil_umbral = np.percentile(importancias_canaritos, (1 - porcentaje_umbral) * 100)
    
    logger.info(f"📊 Umbral de selección: {percentil_umbral:.6f}")
    
    # Seleccionar variables importantes (excluyendo canaritos)
    variables_importantes = {
        feature for feature, importancia in feature_importance_sorted.items()
        if not feature.startswith('canarito') and importancia > percentil_umbral
    }
    
    # Columnas obligatorias
    columnas_obligatorias = {'numero_de_cliente', 'foto_mes', 'clase_ternaria'}
    
    # Combinar sin duplicados
    columnas_finales = columnas_obligatorias.union(variables_importantes)
    
    # Filtrar solo las que existen en el DataFrame
    columnas_existentes = set(df.columns)
    columnas_finales = list(columnas_finales.intersection(columnas_existentes))
    
    # Ordenar: primero obligatorias, luego alfabético
    columnas_ordenadas = []
    for col in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']:
        if col in columnas_finales:
            columnas_ordenadas.append(col)
    
    # Resto de columnas ordenadas
    otras_columnas = sorted([col for col in columnas_finales if col not in columnas_ordenadas])
    columnas_ordenadas.extend(otras_columnas)
    
    logger.info(f"✅ {len(variables_importantes)} variables seleccionadas")
    logger.info(f"📋 {len(columnas_ordenadas)} columnas finales")
    logger.info(f"📈 Dimensionalidad: {df.shape} → {[df.shape[0], len(columnas_ordenadas)]}")
    
    return df.select(columnas_ordenadas)



def calcular_media_limite_compra_polars(df):
    """
    Calcula la media mensual conjunta de los límites de compra de Mastercard y Visa
    usando foto_mes como variable temporal
    """
    return (
        df.lazy()
        .group_by('foto_mes')
        .agg([
            pl.col('Master_mlimitecompra').mean().alias('media_master'),
            pl.col('Visa_mlimitecompra').mean().alias('media_visa'),
            ((pl.col('Master_mlimitecompra') + pl.col('Visa_mlimitecompra')) / 2).mean().alias('media_conjunta')
        ])
        .collect()
    )


def estandarizar_variables_monetarias_polars(df):
    """
    Estandariza variables monetarias usando la media de límites de compra
    REEMPLAZA las columnas originales con las versiones estandarizadas
    """
    # Calcular media mensual
    media_mensual = calcular_media_limite_compra_polars(df)
    
    # Identificar columnas a estandarizar
    columnas_estandarizar = [
        col for col in df.columns 
        if (col.startswith('m') or 
            col.startswith('Visa_m') or 
            col.startswith('Master_m'))
        and col not in ['Master_mlimitecompra', 'Visa_mlimitecompra']
    ]
    
    print(f"Columnas a estandarizar: {len(columnas_estandarizar)}")
    print(f"Ejemplos: {columnas_estandarizar[:5]}...")
    
    # Unir las medias mensuales y aplicar estandarización REEMPLAZANDO las originales
    df_estandarizado = (
        df.lazy()
        .join(media_mensual.lazy(), on='foto_mes', how='left')
        .with_columns([
            (pl.col(col)*10000 / pl.col('media_conjunta')).alias(col)  # Mismo nombre para reemplazar
            for col in columnas_estandarizar
        ])
        .drop(['media_master', 'media_visa', 'media_conjunta'])
        .collect()
    )
    
    return df_estandarizado


def convertir_todo_cero_a_nan(
    df: pl.DataFrame,
    columna_mes: str = "foto_mes"
) -> pl.DataFrame:
    """
    Si toda una columna numérica es cero (o cero/nula) en un mes determinado,
    reemplaza todos sus valores por None (NaN).
    """
    meses_unicos = df[columna_mes].unique().sort()
    resultados = []

    for mes in meses_unicos:
        df_mes = df.filter(pl.col(columna_mes) == mes)
        df_out = df_mes.clone()

        for col in df_mes.columns:
            if col == columna_mes:
                continue
            if df_mes[col].dtype not in [pl.Int64, pl.Float64]:
                continue

            # Verificar si todos los valores son cero o nulos
            todos_cero = df_mes.filter(
                (pl.col(col) != 0) & (pl.col(col).is_not_null())
            ).height == 0

            if todos_cero:
                # Reemplazar todos los valores por None
                df_out = df_out.with_columns(
                    pl.lit(None).alias(col)
                )

        resultados.append(df_out)

    return pl.concat(resultados).sort(columna_mes)



# Intento de arreglo de datadrifting

def ajustar_mediana_6_meses(df, columnas_ajustar, fecha_objetivo=202106, meses_atras=6):
    """
    Ajusta las variables basándose en la mediana de los últimos 6 meses.
    """
    # Calcular fecha límite para los 6 meses hacia atrás
    logger.info(f"Inicio ajuste de mediana a 6 meses sobre el mes {fecha_objetivo}")
    
    fecha_limite = fecha_objetivo - meses_atras
    
    for var in columnas_ajustar:
        if var in df.columns:
            # Mediana histórica (últimos 6 meses excluyendo el objetivo)
            mediana_hist = df.filter(
                (pl.col('foto_mes') >= fecha_limite) & 
                (pl.col('foto_mes') < fecha_objetivo)
            ).select(pl.col(var).median()).item()
            
            # Mediana del mes objetivo
            mediana_objetivo = df.filter(pl.col('foto_mes') == fecha_objetivo).select(
                pl.col(var).median()
            ).item()
            
            # Calcular factor evitando división por cero
            if mediana_objetivo != 0 and mediana_hist is not None:
                factor = mediana_hist / mediana_objetivo
                
                # Aplicar ajuste
                df = df.with_columns(
                    pl.when(pl.col('foto_mes') == fecha_objetivo)
                    .then(pl.col(var) * factor)
                    .otherwise(pl.col(var))
                    .alias(var)
                )
                
    logger.info(f"Fin ajuste de mediana a 6 meses sobre el mes {fecha_objetivo}")
    
    return df



#def reemplazar_columnas_todo_cero_fila(df: pl.DataFrame, columna_mes: str = "foto_mes", columna_id: str = "numero_de_cliente") -> pl.DataFrame:
#    """
#    Si una columna numérica es completamente cero en un mes,
#    reemplaza sus valores fila a fila por el promedio entre el valor del mes anterior y el posterior.
#    """
#    # Ordenar por id y mes
#    df = df.sort([columna_id, columna_mes])
#    meses = sorted(df[columna_mes].unique().to_list())
#
#    columnas_numericas = [
#        c for c, t in zip(df.columns, df.dtypes)
#        if c not in (columna_mes, columna_id) and t in (pl.Float64, pl.Int64)
#    ]
#
#    df_resultado = df.clone()
#
#    for i, mes in enumerate(meses):
#        if i == 0 or i == len(meses) - 1:
#            continue  # No se puede interpolar primer/último mes
#
#        df_mes = df.filter(pl.col(columna_mes) == mes)
#        df_ant = df.filter(pl.col(columna_mes) == meses[i - 1])
#        df_pos = df.filter(pl.col(columna_mes) == meses[i + 1])
#
#        for col in columnas_numericas:
#            # Si toda la columna es cero en este mes
#            if (df_mes[col] == 0).all():
#                # Hacemos join por id con anterior y posterior
#                df_join = (
#                    df_mes.select(columna_id, columna_mes)
#                    .join(df_ant.select([columna_id, col]).rename({col: "valor_ant"}), on=columna_id, how="left")
#                    .join(df_pos.select([columna_id, col]).rename({col: "valor_pos"}), on=columna_id, how="left")
#                    .with_columns(
#                        ((pl.col("valor_ant") + pl.col("valor_pos")) / 2).alias(col)
#                    )
#                    .select([columna_id, col])
#                )
#
#                # Actualizamos en df_resultado los valores del mes correspondiente
#                df_resultado = (
#                    df_resultado.join(df_join, on=columna_id, how="left", suffix="_nuevo")
#                    .with_columns(
#                        pl.when(pl.col(columna_mes) == mes)
#                        .then(pl.col(f"{col}_nuevo"))
#                        .otherwise(pl.col(col))
#                        .alias(col)
#                    )
#                    .drop(f"{col}_nuevo")
#                )
#
#    return df_resultado


#def imputar_con_spline(df: pl.DataFrame, columna_mes: str = "foto_mes", columna_id: str = "numero_de_cliente") -> pl.DataFrame:
#    """
#    Si una columna numérica es completamente cero en un mes,
#    para cada cliente interpola con un spline usando los valores de meses vecinos.
#    """
#    df = df.sort([columna_id, columna_mes])
#    meses = sorted(df[columna_mes].unique().to_list())
#
#    columnas_numericas = [
#        c for c, t in zip(df.columns, df.dtypes)
#        if c not in (columna_mes, columna_id) and t in (pl.Float64, pl.Int64)
#    ]
#
#    df_resultado = df.clone()
#
#    # Detectar qué columnas tienen algún mes con todos ceros
#    columnas_a_tratar = []
#    for col in columnas_numericas:
#        for mes in meses:
#            if (df.filter(pl.col(columna_mes) == mes)[col] == 0).all():
#                columnas_a_tratar.append(col)
#                break  # ya sabemos que hay al menos un mes todo cero
#    columnas_a_tratar = list(set(columnas_a_tratar))
#
#    # Iterar por cliente
#    dfs_imputados = []
#
#    for cliente, df_cliente in df_resultado.group_by(columna_id, maintain_order=True):
#        df_cliente = df_cliente.sort(columna_mes)
#        mes_vals = df_cliente[columna_mes].to_numpy()
#
#        for col in columnas_a_tratar:
#            y = df_cliente[col].to_numpy().astype(float)
#
#            # Si hay valores distintos de cero, podemos ajustar spline
#            mascara_validos = y != 0
#            if mascara_validos.sum() >= 3:  # al menos 3 puntos para spline suave
#                try:
#                    spline = interp.UnivariateSpline(mes_vals[mascara_validos], y[mascara_validos], s=0, k=2)
#                    y_imputado = spline(mes_vals)
#                    # Reemplazar solo donde y == 0
#                    y = np.where(y == 0, y_imputado, y)
#                except Exception:
#                    pass  # si hay algún error numérico, lo dejamos como está
#            df_cliente = df_cliente.with_columns(pl.Series(col, y))
#
#        dfs_imputados.append(df_cliente)
#
#    return pl.concat(dfs_imputados)


def normalizar_clientes_percentil_signo_historico(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estandariza las columnas mensuales por cliente usando el percentil 95 histórico
    (sin mirar hacia el futuro), manteniendo el signo.
    Reemplaza las columnas originales por sus versiones normalizadas.
    """
    
    columnas_mensuales = [c for c in df.columns if c.startswith(("m", "Master_m", "Visa_m"))]
    df = df.sort(["numero_de_cliente", "foto_mes"])

    for col in columnas_mensuales:
        # Percentil 95 histórico acumulado (usa el máximo alcanzado hasta el mes actual)
        df = df.with_columns(
            pl.col(col)
            .abs()
            .quantile(0.95)
            .over("numero_de_cliente")
            .cum_max()
            .alias(f"{col}_p95_hist")
        )

        # Estandarizar respetando el signo
        df = df.with_columns(
            (pl.col(col).sign() * pl.col(col).abs() / pl.col(f"{col}_p95_hist")).alias(col)
        ).drop(f"{col}_p95_hist")
    
    return df

def ajustar_por_inflacion(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajusta las columnas monetarias por inflación según el índice IPC histórico.
    Las columnas afectadas son las que empiezan con 'm', 'Master_m' o 'Visa_m'.
    Convierte automáticamente ints a floats para evitar errores de división.
    """

    ipc_dict = {
        'foto_mes': [
            201901,201902,201903,201904,201905,201906,201907,201908,201909,201910,
            201911,201912,202001,202002,202003,202004,202005,202006,202007,202008,
            202009,202010,202011,202012,202101,202102,202103,202104,202105,202106,
            202107,202108
        ],
        'ipc': [
            1,1.03765622189957,1.04293020255778,1.04820418321598,1.05347816387418,
            1.05875214453239,1.06402612519059,1.06930010584879,1.074574086507,
            1.0798480671652,1.0851220478234,1.09039602848161,1.09567000913981,
            1.10094398979801,1.10621797045622,1.11149195111442,1.11676593177262,
            1.12203991243083,1.12731389308903,1.13258787374723,1.13786185440544,
            1.14313583506364,1.14840981572184,1.15368379638005,1.15895777703825,
            1.16423175769645,1.16950573835466,1.17477971901286,1.18005369967106,
            1.18532768032927,1.19060166098747,1.19587564164567
        ]
    }

    # Especificar esquema explícitamente
    df_ipc = pl.DataFrame(ipc_dict, schema={
        'foto_mes': pl.Int64,
        'ipc': pl.Float64
    })

    # Join por mes
    df = df.join(df_ipc, on="foto_mes", how="left")

    # Columnas a ajustar
    columnas_monetarias = [
        c for c in df.columns if c.startswith(("m", "Master_m", "Visa_m"))
    ]

    # Convertirlas a Float64 primero
    df = df.with_columns([
        pl.col(col).cast(pl.Float64).alias(col) for col in columnas_monetarias
    ])

    # Ajustar por inflación
    df = df.with_columns([
        (pl.col(col) / pl.col("ipc")).alias(col) for col in columnas_monetarias
    ])

    # Eliminar IPC
    df = df.drop("ipc")

    return df

