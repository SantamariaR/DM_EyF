#import pandas as pd
#import duckdb
import logging
import polars as pl
import numpy as np
import lightgbm as lgb
from .config import *
from lightgbm import LGBMRegressor

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
    
    PARAMtrain = {"training": [202011,202012,202101, 202102, 202103]}
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
        pl.when(pl.col("clase_ternaria").is_in(["BAJA+1","BAJA+2"])) # LO CAMBIË PARA EL EXPERIMENTO 53
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


def PPR(dataset: pl.DataFrame, foto_mes_col: str = "foto_mes") -> pl.DataFrame:
    """
    Calcula las combinaciones lineales de Projection Pursuit Regression (PPR)
    usando los pesos obtenidos en R.

    - Escala las variables numéricas por media y desvío estándar dentro de cada `foto_mes`
    - Agrega columnas 'ppr_term_1' y 'ppr_term_2'
    """
    
    logger.info("Inicio del Cálculo de las variables con PPR")

    alpha = {
        "ctrx_quarter": [-0.5451, -0.6825],
        "mcuentas_saldo": [-0.0264, -0.8541],
        "mpasivos_margen": [-0.1729, -0.0899],
        "cdescubierto_preacordado_delta_lag_1": [-0.6106, -0.1276],
        "mrentabilidad_annual_lag_2": [0.0951, 0.3177],
        "Visa_status": [0.2962, -0.5178],
        "cpayroll_trx": [-0.4514, -0.3249],
        "mrentabilidad_annual_lag_1": [-0.1498, -0.8664],
        "Visa_fechaalta_lag_2": [0.1278, -0.0956],
        "Master_mfinanciacion_limite": [0.0197, -0.0956],
    }

    vars_ppr = list(alpha.keys())

    # 1️⃣ Calcular media y desviación por mes
    stats = (
        dataset
        .group_by(foto_mes_col)
        .agg([
            *[pl.col(v).mean().alias(f"{v}_mean") for v in vars_ppr],
            *[pl.col(v).std().alias(f"{v}_std") for v in vars_ppr],
        ])
    )

    # 2️⃣ Unir stats al dataset original (para escalar dentro de cada mes)
    df_joined = dataset.join(stats, on=foto_mes_col, how="left")

    # 3️⃣ Normalizar las variables dentro de cada foto_mes
    df_scaled = df_joined.with_columns([
        ((pl.col(v) - pl.col(f"{v}_mean")) / pl.col(f"{v}_std")).alias(v)
        for v in vars_ppr
    ])

    # 4️⃣ Calcular las combinaciones lineales (αᵗx)
    df_ppr = df_scaled.with_columns([
        sum(pl.col(v) * w[0] for v, w in alpha.items()).alias("ppr_term_1"),
        sum(pl.col(v) * w[1] for v, w in alpha.items()).alias("ppr_term_2")
    ]).select(["ppr_term_1", "ppr_term_2"])

    # 5️⃣ Agregar al dataset original
    dataset_out = dataset.hstack(df_ppr)
    
    logger.info("Fin del Cálculo de las variables con PPR")

    return dataset_out

def agregar_suma_m_visa_master(df: pl.DataFrame) -> pl.DataFrame:
    """
    Agrega una columna 'suma_m_visa_master' con la suma horizontal
    de todas las columnas que comienzan por 'm', 'Visa_m' o 'Master_m'.
    """

    cols = [
        c for c in df.columns
        if c.startswith("m") or c.startswith("Visa_m") or c.startswith("Master_m")
    ]

    if not cols:
        return df

    return df.with_columns(
        pl.sum_horizontal([pl.col(c) for c in cols]).alias("suma_m_visa_master")
    )

def escalar_por_p95_mensual(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reemplaza los valores de todas las columnas numéricas que empiecen por
    m, Visa_m o Master_m dividiéndolos por el valor absoluto del percentil 95
    mensual (agrupado por foto_mes).
    """

    # --- 1. Seleccionar columnas objetivo ---
    patrones = ["^m", "^Visa_m", "^Master_m"]

    cols_objetivo = [
        col for col in df.columns 
        if any(pl.Series([col]).str.contains(pat).to_list()[0] for pat in patrones)
    ]

    if not cols_objetivo:
        raise ValueError("No se encontraron columnas que empiecen por m / Visa_m / Master_m.")

    # --- 2. Calcular p95 mensual en valor absoluto ---
    p95 = (
        df.select(["foto_mes"] + cols_objetivo)
        .group_by("foto_mes")
        .agg([
            pl.col(c).abs().quantile(0.95).alias(f"p95_{c}")
            for c in cols_objetivo
        ])
    )

    # --- 3. Hacer join al dataset original ---
    df2 = df.join(p95, on="foto_mes", how="left")

    # --- 4. Reemplazar cada variable dividiéndola por su p95 absoluto ---
    df2 = df2.with_columns([
        (pl.col(c) / pl.col(f"p95_{c}")).alias(c)
        for c in cols_objetivo
    ])

    # --- 5. Eliminar columnas p95 auxiliares ---
    df2 = df2.drop([f"p95_{c}" for c in cols_objetivo])

    return df2




def entrenar_y_aplicar_quantiles_global(
    df: pl.DataFrame,
    seed=123
):
    """
    Aplicamos quantiles globales a todas las features del dataset sin tener
    en cuenta temporalidad
    """
    logger.info("Empieza el análisis de Quantile Regresion")
    
    clases = ["CONTINUA", "BAJA+1", "BAJA+2"]
    quantiles = [0.10, 0.50, 0.90]
    frac = UNDERSUMPLING
    
    periodos_entrenamiento = MES_TRAIN + [202103] + MES_VALIDACION
    periodo_test = MES_TEST
    
    df = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento+periodo_test)) #FILTRO SOLO LOS MESES DE ENTRENAMIENTO Y TEST

    df_train = df.filter(pl.col("foto_mes").is_in(periodos_entrenamiento))
    
    # UNDERSUMPLIG sobre la clase mayoritaria
    df_pos = df_train.filter(pl.col("clase_ternaria").is_in(["BAJA+1","BAJA+2"]))
    df_neg = df_train.filter(pl.col("clase_ternaria").is_in(["CONTINUA"]))
    
    n_sample = int(df_neg.height * frac)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_train = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0]) 
    df_out = df.clone()

    # detecto features numéricas
    features = [c for c in df.columns if c not in 
                ["foto_mes", "clase_ternaria"]]

    # diccionario: clase → feature → q → modelo
    modelos = {}

    for clase in clases:
        modelos[clase] = {}
        df_c = df_train.filter(pl.col("clase_ternaria") == clase)
        
        logger.info(f"Entrenamineto sobre la clase {clase}")

        for feat in features:
            y = df_c[feat].to_numpy()
            X = np.arange(len(y)).reshape(-1, 1)  # regresión simple temporal
            modelos[clase][feat] = {}

            for q in quantiles:
                params = {
                    "objective": "quantile",
                    "metric": "quantile",
                    "alpha": q,
                    "verbosity": -1,
                    "seed": seed,
                }
                dtrain = lgb.Dataset(X, label=y)
                model = lgb.train(params, dtrain, num_boost_round=80)
                modelos[clase][feat][q] = model

    # --- Aplicamos a TODO el dataset ---
    logger.info("Fin de entrenamientos de los modelos")
    
    N = df.height
    X_full = np.arange(N).reshape(-1, 1)

    logger.info("Aplicamos sobre todo el dataset")
    for clase in clases:
        logger.info(f"Aplicación del modelo de la clase {clase}")
        for feat in features:
            for q in quantiles:
                pred = modelos[clase][feat][q].predict(X_full)
                colname = f"dist_{clase}_q{int(q*100)}_{feat}"
                df_out = df_out.with_columns(
                    (pl.col(feat) - pl.Series(pred)).alias(colname)
                )

    logger.info("Fin de las features de quantiles")
    return df_out


def entrenar_quantiles_rolling(
    df: pl.DataFrame,
    seed=123
):
    logger.info("📌 Iniciando entrenamiento rolling de cuantiles...")


    clases = ["CONTINUA", "BAJA+1", "BAJA+2"]
    quantiles = [0.20, 0.50, 0.80]
    frac = UNDERSUMPLING
    
    meses_ordenados = MES_TRAIN + [202103] + MES_VALIDACION + MES_TEST
    
    df = df.filter(pl.col("foto_mes").is_in(meses_ordenados)) #FILTRO SOLO LOS MESES DE ENTRENAMIENTO Y TEST

    df_train = df.filter(pl.col("foto_mes").is_in(meses_ordenados))
    
    # UNDERSUMPLIG sobre la clase mayoritaria
    df_pos = df_train.filter(pl.col("clase_ternaria").is_in(["BAJA+1","BAJA+2"]))
    df_neg = df_train.filter(pl.col("clase_ternaria").is_in(["CONTINUA"]))
    
    n_sample = int(df_neg.height * frac)
    df_neg = df_neg.sample(n=n_sample, seed=SEMILLA[0])

    # Concatenar positivos y negativos muestreados
    df_sub = pl.concat([df_pos, df_neg])

    # Shuffle del dataset
    df_train = df_sub.sample(fraction=1.0, shuffle=True, seed=SEMILLA[0]) 
    df_out = df.clone()

    # detectar features numéricas
    features = [c for c in df.columns if c not in 
                ["foto_mes", "clase_ternaria"]]

    logger.info(f"🔍 Features detectadas: {len(features)}")
    logger.info(f"🔍 Clases: {clases}")
    logger.info(f"🔍 Quantiles: {quantiles}")

    # ============================================================
    # 1) Crear TODAS las columnas vacías sólo una vez
    # ============================================================
    logger.info("🛠 Creando columnas vacías para distancias...")

    all_cols = []
    for clase in clases:
        for feat in features:
            for q in quantiles:
                col = f"dist_{clase}_q{int(q*100)}_{feat}"
                all_cols.append(pl.lit(None).alias(col))

    df_out = df_out.with_columns(all_cols)
    logger.info("   ✔ Columnas creadas.")

    # ============================================================
    # 2) Loop principal por mes
    # ============================================================
    for mes_actual in meses_ordenados:

        logger.info(f"\n📅 === Procesando mes: {mes_actual} ===")

        df_train = df.filter(pl.col("foto_mes") < mes_actual)
        df_mes = df.filter(pl.col("foto_mes") == mes_actual)

        logger.info(f"     → Entrenamiento: {df_train.height} filas")
        logger.info(f"     → Filas del mes actual: {df_mes.height}")

        if df_train.is_empty():
            logger.info("     ⚠ Sin datos para entrenar este mes → columnas quedan NaN")
            continue

        N_mes = df_mes.height
        X_mes = np.arange(N_mes).reshape(-1, 1)

        updates = {}

        # Entrenamiento por clase
        for clase in clases:
            logger.info(f"   ▶ Entrenando clase '{clase}'...")

            df_c = df_train.filter(pl.col("clase_ternaria") == clase)

            if df_c.is_empty():
                logger.info("     ⚠ Esta clase no tiene registros en el training.")
                continue

            y_df_c = df_c.select(features)
            n_train = df_c.height
            X = np.arange(n_train).reshape(-1, 1)

            # Entrenamiento por feature
            for feat in features:
                y = y_df_c[feat].to_numpy()

                # Entrenamiento por cuantiles
                for q in quantiles:
                    logger.info(f"      → Entrenando {feat} / clase {clase} / q={q}")

                    params = {
                        "objective": "quantile",
                        "metric": "quantile",
                        "alpha": q,
                        "verbosity": -1,
                        "seed": seed
                    }

                    model = lgb.train(
                        params,
                        lgb.Dataset(X, label=y),
                        num_boost_round=80
                    )

                    pred_mes = model.predict(X_mes)
                    colname = f"dist_{clase}_q{int(q*100)}_{feat}"

                    updates[colname] = (
                        df_mes[feat].to_numpy() - pred_mes
                    )

        # ======================================================
        # 3) Aplicar actualizaciones *en bloque*
        # ======================================================
        if updates:
            logger.info(f"📝 Actualizando {len(updates)} columnas para el mes {mes_actual}...")

            df_out = df_out.with_columns([
                pl.when(pl.col("foto_mes") == mes_actual)
                 .then(pl.Series(name, vals))
                 .otherwise(pl.col(name))
                 .alias(name)
                for name, vals in updates.items()
            ])

            logger.info("   ✔ Columnas actualizadas.")
        else:
            logger.info("   ⚠ Ninguna columna fue actualizada para este mes.")

    logger.info("\n🎉 Finalizado el entrenamiento rolling de cuantiles.")
    return df_out