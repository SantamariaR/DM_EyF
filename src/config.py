import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["competencia01"]

        STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "Wendsday")
        DATA_PATH = _cfg.get("DATA_PATH", "../content/datasets/competencia_01.csv.gz")
        DATA_PATH_LOCAL = _cfg.get("DATA_PATH_LOCAL", "../data/competencia_01_crudo.csv")
        SEMILLA = _cfg.get("SEMILLA", [42])
        BUCKET_NAME = _cfgGeneral.get("BUCKET_NAME", "content/buckets/b1/exp")
        MES_TRAIN = _cfg.get("MES_TRAIN", 202102)
        MES_VALIDACION = _cfg.get("MES_VALIDACION", 202103)
        MES_TEST = _cfg.get("MES_TEST", 202104)
        MES_PREDIC = _cfg.get("MES_PREDIC", 202106)
        GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
        COSTO_ESTIMULO = _cfg.get("COSTO_ESTIMULO", None)
        INICIO_ENVIOS = _cfg.get("INICIO_ENVIOS", None)
        FIN_ENVIOS = _cfg.get("FIN_ENVIOS", None)
        PASO_ENVIOS = _cfg.get("PASO_ENVIOS", None)
        UNDERSUMPLING = _cfg.get("UNDERSUMPLING", 0.5)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise
