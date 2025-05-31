"""
Configuracion global del proyecto de mineria de datos para trading de criptomonedas
"""

# Configuracion de la API de Basescan
API_KEY = "XVUW2UD38TQTFW14AJF5X7JMPNNRTYZGAI"  # Reemplazar con tu clave API de Basescan
API_URL = "https://basescan.org/api"
API_RATE_LIMIT = 5  # Solicitudes por segundo (ajustar segun documentacion de Basescan)

# Rutas de archivos y directorios
import os

# Obtenemos la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, "datos")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Asegurarse que existen los directorios
os.makedirs(DATOS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Archivos de datos
DATOS_CRUDOS = os.path.join(DATOS_DIR, "bitcoin_raw.csv")
DATOS_PROCESADOS = os.path.join(DATOS_DIR, "bitcoin_procesado.csv")
DATOS_DISCRETIZADOS = os.path.join(DATOS_DIR, "bitcoin_discretizado.csv")
MODELO_CLUSTERING = os.path.join(DATOS_DIR, "modelo_clustering.pkl")
MODELO_ANOMALIAS = os.path.join(DATOS_DIR, "modelo_anomalias.pkl")
REGLAS_ASOCIACION = os.path.join(DATOS_DIR, "reglas_asociacion.csv")

# Parametros de los modelos
PARAMETROS = {
    # Parametros para la deteccion de anomalias
    "anomalia_umbral_retorno": 3.0,  # % de cambio considerado anomalia (reducido de 5.0)
    "anomalia_umbral_volumen": 1.5,  # Desviaciones estandar sobre la media (reducido de 2.0)
    
    # Parametros para clustering
    "clustering_num_clusters": 4,
    "clustering_random_state": 42,
    
    # Parametros para random forest
    "rf_num_arboles": 100,
    "rf_max_depth": 10,
    "rf_random_state": 42,
    
    # Parametros para reglas de asociacion
    "reglas_soporte_min": 0.1,
    "reglas_confianza_min": 0.7,
    "reglas_lift_min": 1.2,
    
    # Parametros para indicadores tecnicos
    "ventana_rsi": 14,
    "ventana_macd_rapida": 12,
    "ventana_macd_lenta": 26,
    "ventana_macd_senal": 9,
    "ventana_volatilidad": 7,
    "ventana_media_movil": [20, 50, 200],  # Dias para medias moviles
    
    # Parametros para backtesting
    "capital_inicial": 10000,  # USD
    "comision": 0.1,  # % por operacion
}

# Configuracion de registros (logs)
import logging

# Configuracion de logging
LOG_FILE = os.path.join(LOGS_DIR, "mineria_trading.log")
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configuracion de la interfaz grafica
GUI_CONFIG = {
    "ventana_titulo": "Sistema de Mineria de Datos para Trading de Bitcoin",
    "ventana_ancho": 1200,
    "ventana_alto": 800,
    "grafico_alto": 500,
    "tema": "claro",  # 'claro' o 'oscuro'
    "colores": {
        "fondo": "#f5f5f5",
        "texto": "#333333",
        "primario": "#1976D2",
        "secundario": "#388E3C",
        "alerta": "#D32F2F",
        "exito": "#43A047",
        "cluster_colores": ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"]
    }
}

# Fechas para la extraccion de datos
from datetime import datetime, timedelta

FECHA_FIN = datetime.now()
FECHA_INICIO = FECHA_FIN - timedelta(days=365*5)  # 5 años de datos

# Modo debug (para desarrollo)
DEBUG = True