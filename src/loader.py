import polars as pl
import logging

# Carpeta que contiene las funciones carga y las operaciones sobre la clase ternaria


logger = logging.getLogger("__name__")

## Funcion para cargar datos
def cargar_datos(path: str) -> pl.DataFrame | None:

    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pl.read_csv(path, infer_schema_length=1000000)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise
    
# Calculamos la clase_ternaria
def calcular_clase_ternaria(dataset: pl.DataFrame) -> pl.DataFrame:
    """
    Versión optimizada del cálculo de clase_ternaria.
    """
    
    # Calcular periodo0 y ordenar
    dsimple = (
        dataset
        .with_columns(
            pl.col("foto_mes").alias("pos"),
            (pl.col("foto_mes").floordiv(100) * 12 + pl.col("foto_mes").mod(100)).alias("periodo0")
        )
        .select(["pos", "numero_de_cliente", "periodo0"])
        .sort(["numero_de_cliente", "periodo0"])
    )
    
    # Calcular periodos de referencia
    periodo_ultimo = dsimple["periodo0"].max()
    periodo_anteultimo = periodo_ultimo - 1
    
    # Calcular leads y clase_ternaria en una secuencia
    dsimple = (
        dsimple
        .with_columns([
            pl.col("periodo0").shift(-1).over("numero_de_cliente").alias("periodo1"),
            pl.col("periodo0").shift(-2).over("numero_de_cliente").alias("periodo2")
        ])
        .with_columns(
            # Inicializar con CONTINUA
            pl.when(pl.col("periodo0") < periodo_anteultimo)
            .then(pl.lit("CONTINUA"))
            .otherwise(pl.lit(None))
            .alias("clase_ternaria")
        )
        .with_columns(
            # Aplicar BAJA+1
            pl.when(
                (pl.col("periodo0") < periodo_ultimo) &
                (pl.col("periodo1").is_null() | (pl.col("periodo0") + 1 < pl.col("periodo1")))
            )
            .then(pl.lit("BAJA+1"))
            .otherwise(pl.col("clase_ternaria"))
            .alias("clase_ternaria")
        )
        .with_columns(
            # Aplicar BAJA+2
            pl.when(
                (pl.col("periodo0") < periodo_anteultimo) &
                (pl.col("periodo0") + 1 == pl.col("periodo1")) &
                (pl.col("periodo2").is_null() | (pl.col("periodo0") + 2 < pl.col("periodo2")))
            )
            .then(pl.lit("BAJA+2"))
            .otherwise(pl.col("clase_ternaria"))
            .alias("clase_ternaria")
        )
        .sort("pos")  # Reordenar por posición original
    )
    
    # Unir al dataset original
    return dataset.with_columns(dsimple.select("clase_ternaria"))


# Chequeo de conteo
def contar_por_grupos(dataset: pl.DataFrame) -> pl.DataFrame:
    """
    Réplica exacta del código R:
    setorder(dataset, foto_mes, clase_ternaria, numero_de_cliente)
    dataset[, .N, list(foto_mes, clase_ternaria)]
    """
    # Ordenar
    dataset_ordenado = dataset.sort(["foto_mes", "clase_ternaria", "numero_de_cliente"])
    
    # Contar por grupos
    resultado = dataset_ordenado.group_by(["foto_mes", "clase_ternaria"]).agg(
        pl.count().alias("N")
    )
    
    return resultado


def convertir_clase_ternaria_a_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    - BAJA+1 y BAJA+2 = 1
  
    Args:
        df: DataFrame de Polars con columna 'clase_ternaria'
  
    Returns:
        pl.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """
    df_result = df.clone()
    
    # Contar valores originales de forma más segura
    counts_dict = df_result['clase_ternaria'].value_counts()
    
    # Usar get para evitar errores si algún valor no existe
    n_continua_orig = counts_dict.filter(pl.col('clase_ternaria') == 'CONTINUA').select('count').item()
    n_baja1_orig = counts_dict.filter(pl.col('clase_ternaria') == 'BAJA+1').select('count').item()
    n_baja2_orig = counts_dict.filter(pl.col('clase_ternaria') == 'BAJA+2').select('count').item()

    
    # Crear nueva columna clase_01 - FORZANDO TIPO NUMÉRICO
    df_result = df_result.with_columns(
        pl.when(pl.col('clase_ternaria') == 'CONTINUA')
        .then(pl.lit(0))  # Usar pl.lit() para forzar tipo numérico
        .when(pl.col('clase_ternaria').is_in(['BAJA+1', 'BAJA+2']))
        .then(pl.lit(1))  # Usar pl.lit() para forzar tipo numérico
        .alias('clase_01')
    )
    
    
    # Log de la conversión
    n_ceros = df_result.filter(pl.col('clase_01') == 0).height
    n_unos = df_result.filter(pl.col('clase_01') == 1).height
  
    logger.info(f"Conversión completada:")
    logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")
  
    return df_result


def cargar_features_importantes(path: str,ordenar_por_importancia=True) -> list:

    '''
    Carga un CSV desde 'path' y retorna un dict con las features importance.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pl.read_csv(path, infer_schema_length=100)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        # Ordenar por importancia si se solicita
        if ordenar_por_importancia and 'importance' in df.columns:
            df = df.sort('importance', descending=True)
            feature_dict = dict(
                    df.select([
                    pl.col('Feature'),
                    pl.col('Importance')
                 ]).iter_rows()
            )
        return feature_dict
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise   