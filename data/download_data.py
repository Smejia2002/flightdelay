import pandas as pd
import os
import sys

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
# URL directa(raw pointer al archivo parquet)
DATASET_URL = "https://huggingface.co/datasets/mejiadev7/flight_delay/resolve/main/dataset_prepared.parquet"

# Ruta local donde guardaremos el archivo para que el modelo lo consuma
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "dataset_prepared.parquet"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

def download_data():

    print("🚀 Iniciando proceso de descarga de datos...")
    
    # 1. Crear el directorio si no existe 
    if not os.path.exists(OUTPUT_DIR):
        print(f"📂 Creando directorio: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Verificar si el archivo ya existe
    if os.path.exists(OUTPUT_PATH):
        print(f"⚠️ El archivo ya existe en: {OUTPUT_PATH}")
        response = input("¿Quieres descargarlo de nuevo y sobreescribirlo? (s/n): ")
        if response.lower() != 's':
            print("⏭️  Salto de descarga. Usando archivo existente.")
            return

    # 3. Descarga
    print(f"⬇️  Descargando desde: {DATASET_URL}")
    try:
        # Pandas maneja la conexión HTTPS y la lectura del parquet automáticamente
        df = pd.read_parquet(DATASET_URL)
        
        # Guardar localmente
        print(f"💾 Guardando en disco: {OUTPUT_PATH}...")
        df.to_parquet(OUTPUT_PATH, index=False)
        
        print(f"✅ ¡Éxito! Dataset listo para usarse.")
        print(f"📊 Info: {df.shape[0]} filas, {df.shape[1]} columnas.")
        
    except Exception as e:
        print(f"❌ Error crítico durante la descarga: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_data()
