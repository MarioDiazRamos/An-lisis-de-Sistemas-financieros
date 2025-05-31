"""
Script para probar la conexión a la API de Basescan.org y diagnóstico de problemas
"""

import requests
import time
import logging
from datetime import datetime, timedelta
import sys
import json

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_test.log')
    ]
)
logger = logging.getLogger('api_test')

# Parámetros para la API
API_KEY = "XVUW2UD38TQTFW14AJF5X7JMPNNRTYZGAI"  # Usamos la clave de config.py
API_URL = "https://basescan.org/api"

# URLs alternativas para probar
ALTERNATIVE_URLS = [
    "https://api.basescan.org",
    "https://api.base.org",
    "https://api-basescan.org",
]

def test_api_connection():
    """
    Prueba básica de conexión a la API
    """
    logger.info("Probando conexión básica a la API de Basescan...")
    
    try:
        response = requests.get(f"{API_URL}/ping", timeout=5)
        logger.info(f"Resultado de ping: Status Code = {response.status_code}")
        logger.info(f"Respuesta: {response.text[:200]}")  # Mostramos solo los primeros 200 caracteres
        
        if response.status_code == 200:
            logger.info("✅ Conexión básica exitosa")
        else:
            logger.warning(f"⚠️ La conexión devolvió un código de estado inesperado: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error de conexión: {str(e)}")

def test_api_key():
    """
    Prueba la validez de la clave API
    """
    logger.info("Probando validez de la clave API...")
    
    # Endpoint que requiere autenticación
    endpoint = f"{API_URL}/v1/ticker/price"
    
    params = {
        'apikey': API_KEY,
        'symbol': 'BTCUSDT'
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=5)
        logger.info(f"Status Code = {response.status_code}")
        
        if response.status_code == 200:
            logger.info(f"✅ Clave API válida: {response.text[:200]}")
        elif response.status_code == 401 or response.status_code == 403:
            logger.error(f"❌ Clave API inválida o sin permisos: {response.text[:200]}")
        else:
            logger.warning(f"⚠️ Respuesta inesperada: {response.status_code} - {response.text[:200]}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error al probar clave API: {str(e)}")

def test_endpoints():
    """
    Prueba diferentes endpoints de la API para verificar su funcionamiento
    """
    logger.info("Probando endpoints comunes de la API...")
    
    endpoints = [
        ("/v1/ticker/price", {'symbol': 'BTCUSDT'}),
        ("/v1/ticker/24hr", {'symbol': 'BTCUSDT'}),
        ("/v1/klines", {'symbol': 'BTCUSDT', 'interval': '1d', 'limit': 10})
    ]
    
    for endpoint, params in endpoints:
        logger.info(f"Probando endpoint: {endpoint}")
        params['apikey'] = API_KEY
        
        try:
            response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
            logger.info(f"Status Code = {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"✅ Endpoint {endpoint} funcionando correctamente")
                try:
                    # Intentar parsear la respuesta como JSON para ver la estructura
                    json_data = response.json()
                    logger.info(f"Estructura de respuesta: {json.dumps(json_data, indent=2)[:300]}...")
                except:
                    logger.info(f"Respuesta (no JSON): {response.text[:200]}")
            else:
                logger.warning(f"⚠️ El endpoint {endpoint} devolvió: {response.status_code} - {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error al probar endpoint {endpoint}: {str(e)}")
        
        # Pausa para evitar límites de tasa
        time.sleep(2)

def test_alternative_urls():
    """
    Prueba URLs alternativas para la API
    """
    logger.info("Probando URLs alternativas...")
    
    for url in ALTERNATIVE_URLS:
        logger.info(f"Probando URL alternativa: {url}")
        
        try:
            # Intentar un endpoint simple que no requiera autenticación si es posible
            response = requests.get(f"{url}/ping", timeout=5)
            logger.info(f"Status Code = {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"✅ URL alternativa {url} parece funcionar")
            else:
                logger.warning(f"⚠️ URL alternativa {url} devolvió: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error con URL alternativa {url}: {str(e)}")
        
        time.sleep(1)

def test_historical_data():
    """
    Prueba la extracción de datos históricos (caso de uso principal)
    """
    logger.info("Probando extracción de datos históricos...")
    
    # Definir fechas para la prueba
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=7)  # Probar con solo una semana
    
    # Convertir a timestamps
    inicio_ts = int(fecha_inicio.timestamp())
    fin_ts = int(fecha_fin.timestamp())
    
    # Parámetros para la solicitud
    endpoint = f"{API_URL}/v1/klines"
    params = {
        'apikey': API_KEY,
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': inicio_ts * 1000,  # Convertir a milisegundos
        'endTime': fin_ts * 1000,
        'limit': 10
    }
    
    try:
        logger.info(f"Solicitando datos históricos desde {fecha_inicio} hasta {fecha_fin}")
        response = requests.get(endpoint, params=params, timeout=10)
        logger.info(f"Status Code = {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                logger.info(f"✅ Datos históricos recibidos: {len(data)} registros")
                # Mostrar el primer registro para inspección
                logger.info(f"Primer registro: {data[0]}")
            else:
                logger.warning(f"⚠️ No se recibieron datos históricos o formato inesperado: {data}")
        else:
            logger.error(f"❌ Error al obtener datos históricos: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error de solicitud: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error inesperado: {str(e)}")

def check_api_documentation():
    """
    Verifica si la URL de documentación de la API es accesible
    """
    logger.info("Verificando documentación de la API...")
    
    # URLs típicas de documentación
    doc_urls = [
        "https://basescan.org/api-docs",
        "https://docs.basescan.org",
        "https://basescan.org/docs",
        "https://api.basescan.org/docs"
    ]
    
    for url in doc_urls:
        try:
            response = requests.get(url, timeout=5)
            logger.info(f"URL de documentación {url}: Status Code = {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"✅ Documentación disponible en: {url}")
            else:
                logger.info(f"⚠️ Documentación no disponible en: {url}")
        except requests.exceptions.RequestException:
            logger.info(f"❌ No se pudo acceder a la documentación en: {url}")

def main():
    """
    Función principal que ejecuta todas las pruebas
    """
    logger.info("=== DIAGNÓSTICO DE API BASESCAN ===")
    logger.info(f"URL de la API: {API_URL}")
    logger.info(f"Clave API (3 primeros caracteres): {API_KEY[:3]}{'*' * 10}")
    
    test_api_connection()
    test_api_key()
    test_endpoints()
    test_alternative_urls()
    test_historical_data()
    check_api_documentation()
    
    logger.info("=== FIN DEL DIAGNÓSTICO ===")
    logger.info("Revise el archivo api_test.log para detalles completos")

if __name__ == "__main__":
    main()
