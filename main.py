import pandas as pd
from datetime import datetime
import os


def main():
    print(">>> Inicio de ejecución")

    # Asegurar que exista la carpeta de logs
    os.makedirs("logs", exist_ok=True)

    # Cargar dataset desde carpeta data
    try:
        df = pd.read_csv("data/competencia_01_crudo.csv")
    except FileNotFoundError:
        print("No se encontró el archivo data/competencia_01_crudo.csv")
        return

    # Mostrar primeras filas en consola
    print(df.head())

    # Información básica
    filas, columnas = df.shape
    mensaje = f"[{datetime.now()}] Dataset cargado con {filas} filas y {columnas} columnas\n"

    # Guardar log en archivo
    with open("logs/logs.txt", "a", encoding="utf-8") as f:
        f.write(mensaje)

    print(">>> Ejecución finalizada. Revisa logs/logs.txt")


if __name__ == "__main__":
    main()
