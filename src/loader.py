import polars as pl
import logging

logger = logging.getLogger("__name__")

## Funcion para cargar datos
def cargar_datos(path: str) -> pl.DataFrame | None:

    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pl.read_csv(path, infer_schema_length=30000)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise
    
def convertir_a_clase_ternaria(df: pl.DataFrame, columna: str) -> pl.DataFrame:

    return df


# Calcuñlamos la clase_ternaria
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
