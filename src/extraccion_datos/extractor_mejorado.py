"""
Modulo mejorado para la extraccion de datos historicos de criptomonedas desde API de Basescan
"""

import os
import time
import logging
import requests
import pandas as pd
import json
from datetime import datetime, timedelta

class Extractor:
    """
    Clase para la extraccion de datos historicos de criptomonedas
    """
    
    def __init__(self, api_key, api_url, rate_limit=5, timeout=30, max_retries=3):
        """
        Inicializa el extractor de datos
        
        Args:
            api_key (str): Clave de API para Basescan
            api_url (str): URL base de la API
            rate_limit (int): Límite de solicitudes por segundo
            timeout (int): Tiempo máximo de espera para solicitudes en segundos
            max_retries (int): Número máximo de reintentos para solicitudes fallidas
        """
        self.api_key = api_key
        self.api_url = api_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()  # Usar una sesión para mejor rendimiento
        
        # Configurar headers comunes para las solicitudes
        self.session.headers.update({
            'User-Agent': 'MineriaDatosApp/1.0',
            'Accept': 'application/json'
        })
        
        self.logger = logging.getLogger(__name__)
    
    def _make_api_request(self, endpoint, params, retry_count=0):
        """
        Realiza una solicitud a la API con manejo de errores y reintentos
        
        Args:
            endpoint (str): Endpoint de la API
            params (dict): Parámetros para la solicitud
            retry_count (int): Contador de reintentos
            
        Returns:
            dict or list: Datos de respuesta de la API
        """
        full_url = f"{self.api_url}{endpoint}"
        params['apikey'] = self.api_key
        
        try:
            # Aplicar limitación de tasa
            time.sleep(1.0 / self.rate_limit)
            
            # Realizar la solicitud
            self.logger.debug(f"Solicitando {full_url} con parámetros: {params}")
            response = self.session.get(full_url, params=params, timeout=self.timeout)
            
            # Registrar detalles de la respuesta para diagnóstico
            self.logger.debug(f"Estado de respuesta: {response.status_code}, Longitud: {len(response.content)}")
            
            # Verificar si la respuesta es exitosa
            response.raise_for_status()
            
            # Intentar decodificar la respuesta como JSON
            data = response.json()
            return data
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else "desconocido"
            
            # Registrar información detallada sobre el error
            self.logger.error(f"Error HTTP {status_code} en solicitud a {full_url}")
            self.logger.error(f"Parámetros: {params}")
            
            if hasattr(e, 'response'):
                self.logger.error(f"Respuesta de error: {e.response.text[:1000]}")
            
            # Errores específicos
            if status_code == 403:
                self.logger.error("Error 403 Forbidden: Problemas con la autenticación de la API key")
                self.logger.info("Sugerencias: 1) Verificar que la API key es correcta, 2) Revisar permisos, 3) Comprobar restricciones IP")
            elif status_code == 429:
                self.logger.warning("Error 429 Too Many Requests: Límite de tasa excedido")
                if retry_count < self.max_retries:
                    wait_time = min(30, 2 ** retry_count)  # Espera exponencial
                    self.logger.info(f"Esperando {wait_time} segundos antes de reintentar...")
                    time.sleep(wait_time)
                    return self._make_api_request(endpoint, params, retry_count + 1)
            
            # Reintentar con errores 5xx
            if status_code >= 500 and retry_count < self.max_retries:
                wait_time = min(30, 2 ** retry_count)
                self.logger.info(f"Error del servidor, reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
                return self._make_api_request(endpoint, params, retry_count + 1)
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Error de conexión: {str(e)}")
            if retry_count < self.max_retries:
                wait_time = min(30, 2 ** retry_count)
                self.logger.info(f"Problema de conexión, reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
                return self._make_api_request(endpoint, params, retry_count + 1)
                
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout en la solicitud: {str(e)}")
            if retry_count < self.max_retries:
                wait_time = min(30, 2 ** retry_count)
                self.logger.info(f"Tiempo de espera agotado, reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
                return self._make_api_request(endpoint, params, retry_count + 1)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Error al decodificar respuesta JSON: {str(e)}")
            self.logger.error(f"Respuesta recibida: {response.text[:200]}...")
            
        except Exception as e:
            self.logger.error(f"Error inesperado: {str(e)}", exc_info=True)
            
        # Si llegamos aquí, todos los intentos fallaron
        if retry_count >= self.max_retries:
            self.logger.error(f"Se alcanzó el número máximo de reintentos ({self.max_retries})")
        
        return None
        
    def extraer_datos_historicos(self, fecha_inicio, fecha_fin, divisa="BTC"):
        """
        Extrae datos historicos de precio y volumen para una criptomoneda
        
        Args:
            fecha_inicio (datetime): Fecha de inicio para los datos
            fecha_fin (datetime): Fecha final para los datos
            divisa (str): Simbolo de la criptomoneda (default: "BTC")
            
        Returns:
            pd.DataFrame: DataFrame con los datos historicos
        """
        self.logger.info(f"Extrayendo datos históricos para {divisa} desde {fecha_inicio} hasta {fecha_fin}")
        
        # Convertir fechas a formato timestamp para la API
        inicio_ts = int(fecha_inicio.timestamp())
        fin_ts = int(fecha_fin.timestamp())
        
        try:
            # Verificar si hay una clave API válida
            if self.api_key and self.api_key != "TU_CLAVE_API_AQUI":
                # Intentar primero con Basescan.org
                datos_basescan = self._extraer_datos_basescan(inicio_ts, fin_ts, divisa)
                
                if datos_basescan and len(datos_basescan) > 0:
                    self.logger.info(f"Datos extraídos con éxito de Basescan: {len(datos_basescan)} registros")
                    return self._crear_dataframe(datos_basescan)
                
                # Si Basescan falla, intentar con API alternativa
                datos_alternativos = self._extraer_datos_alternativos(inicio_ts, fin_ts, divisa)
                
                if datos_alternativos and len(datos_alternativos) > 0:
                    self.logger.info(f"Datos extraídos de API alternativa: {len(datos_alternativos)} registros")
                    return self._crear_dataframe(datos_alternativos)
                    
                # Si ambas APIs fallan, generar datos sintéticos
                self.logger.warning("Todas las APIs fallaron, generando datos sintéticos")
                datos_sinteticos = self._generar_datos_sinteticos(fecha_inicio, fecha_fin)
                return self._crear_dataframe(datos_sinteticos)
            else:
                # Si no hay clave API válida, usar datos sintéticos
                self.logger.warning("Clave API no válida o no proporcionada, generando datos sintéticos")
                datos_sinteticos = self._generar_datos_sinteticos(fecha_inicio, fecha_fin)
                return self._crear_dataframe(datos_sinteticos)
                
        except Exception as e:
            self.logger.error(f"Error al extraer datos: {str(e)}", exc_info=True)
            # En caso de error, generar datos sintéticos
            self.logger.warning("Generando datos sintéticos debido a errores")
            datos_sinteticos = self._generar_datos_sinteticos(fecha_inicio, fecha_fin)
            return self._crear_dataframe(datos_sinteticos)
    
    def _extraer_datos_basescan(self, inicio_ts, fin_ts, divisa):
        """
        Extrae datos históricos desde la API de Basescan
        
        Args:
            inicio_ts (int): Timestamp de inicio
            fin_ts (int): Timestamp de fin
            divisa (str): Símbolo de la criptomoneda
            
        Returns:
            list: Lista de datos históricos
        """
        self.logger.info(f"Intentando extracción de datos desde Basescan para {divisa}")
        
        endpoint = "/v1/klines"
        
        data = []
        current_start = inicio_ts * 1000
        
        # Hacer múltiples llamadas si el período es grande (paginación)
        while current_start < fin_ts * 1000:
            params = {
                'symbol': f"{divisa}USDT",
                'interval': '1d',
                'startTime': current_start,
                'endTime': fin_ts * 1000,
                'limit': 1000  # Máximo número de registros
            }
            
            batch_data = self._make_api_request(endpoint, params)
            
            if not batch_data:
                self.logger.warning(f"No se obtuvieron datos para el período {datetime.fromtimestamp(current_start/1000)}")
                break
            
            # Convertir datos de la API a nuestro formato
            for item in batch_data:
                try:
                    timestamp_ms, open_price, high, low, close, volume, close_time, *_ = item
                    
                    data.append({
                        'timestamp': timestamp_ms // 1000,  # Convertir a segundos
                        'open': float(open_price),
                        'high': float(high),
                        'low': float(low),
                        'close': float(close),
                        'volume': float(volume),
                        'market_cap': float(close) * float(volume)  # Estimación
                    })
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Error procesando item de datos: {str(e)} - Item: {item}")
            
            if len(batch_data) < 1000:
                break
            
            # Establecer siguiente punto de inicio
            current_start = batch_data[-1][0] + 1
        
        return data
    
    def _extraer_datos_alternativos(self, inicio_ts, fin_ts, divisa):
        """
        Extrae datos desde una API alternativa cuando Basescan falla
        
        Args:
            inicio_ts (int): Timestamp de inicio
            fin_ts (int): Timestamp de fin
            divisa (str): Símbolo de la criptomoneda
            
        Returns:
            list: Lista de datos históricos o None si falla
        """
        # Ejemplo: Usar CoinGecko como alternativa (no requiere API key)
        # Nota: Implementación simplificada, debe adaptarse según la API alternativa
        self.logger.info(f"Intentando extracción de datos desde API alternativa para {divisa}")
        
        try:
            # Convertir timestamps a formato para API alternativa
            from_date = datetime.fromtimestamp(inicio_ts).strftime('%d-%m-%Y')
            to_date = datetime.fromtimestamp(fin_ts).strftime('%d-%m-%Y')
            
            # URL de ejemplo para API alternativa (CoinGecko)
            url = f"https://api.coingecko.com/api/v3/coins/{divisa.lower()}/market_chart"
            params = {
                'vs_currency': 'usd',
                'from': inicio_ts,
                'to': fin_ts,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            json_data = response.json()
            
            # Procesar datos según formato API alternativa
            if 'prices' in json_data and len(json_data['prices']) > 0:
                data = []
                
                # Ejemplo de procesamiento para CoinGecko
                for i, (timestamp_ms, price) in enumerate(json_data['prices']):
                    volume = json_data['total_volumes'][i][1] if i < len(json_data['total_volumes']) else 0
                    
                    data.append({
                        'timestamp': timestamp_ms // 1000,  # ms a segundos
                        'open': price,  # Aproximación
                        'high': price,  # Aproximación
                        'low': price,  # Aproximación
                        'close': price,
                        'volume': volume,
                        'market_cap': price * volume  # Estimación
                    })
                
                return data
            
        except Exception as e:
            self.logger.error(f"Error al extraer datos desde API alternativa: {str(e)}")
        
        return None
    
    def _crear_dataframe(self, datos):
        """
        Convierte los datos en un DataFrame estructurado
        
        Args:
            datos (list): Lista de diccionarios con datos
            
        Returns:
            pd.DataFrame: DataFrame con los datos procesados
        """
        # Crear DataFrame a partir de los datos
        df = pd.DataFrame(datos)
        
        # Convertir la columna de timestamp a datetime
        df['fecha'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('fecha')
        
        # Ordenar por fecha ascendente
        df = df.sort_index()
        
        return df
    
    def _generar_datos_sinteticos(self, fecha_inicio, fecha_fin):
        """
        Genera datos sinteticos para desarrollo y pruebas
        
        Args:
            fecha_inicio (datetime): Fecha de inicio
            fecha_fin (datetime): Fecha fin
            
        Returns:
            list: Lista de diccionarios con datos sintéticos
        """
        import numpy as np
        
        self.logger.warning("Generando datos sintéticos. En producción, use la API real.")
        
        # Crear rango de fechas
        dias = (fecha_fin - fecha_inicio).days + 1
        fechas = [fecha_inicio + timedelta(days=i) for i in range(dias)]
        timestamps = [int(fecha.timestamp()) for fecha in fechas]
        
        # Precio inicial
        precio_base = 40000  # BTC precio base
        
        # Generar precios con tendencia alcista y volatilidad realista
        np.random.seed(42)  # Para reproducibilidad
        
        # Simular cambios diarios con volatilidad del 3% (realista para BTC)
        cambios_diarios = np.random.normal(0.001, 0.03, dias)  # Media ligeramente positiva
        
        # Simular algunas tendencias y volatilidad estacional
        tendencia = np.linspace(0, 0.3, dias)  # Tendencia alcista general
        ciclo = 0.1 * np.sin(np.linspace(0, 4*np.pi, dias))  # Ciclos
        
        # Combinar efectos
        cambios = cambios_diarios + 0.001 * tendencia + ciclo
        
        # Calcular precios acumulativos
        precios_relativos = np.cumprod(1 + cambios)
        precios = precio_base * precios_relativos
        
        # Generar datos simulados para cada día
        data = []
        for i, ts in enumerate(timestamps):
            precio = precios[i]
            high = precio * (1 + abs(np.random.normal(0, 0.01)))
            low = precio * (1 - abs(np.random.normal(0, 0.01)))
            open_price = precio * (1 + np.random.normal(0, 0.005))
            
            # Volumen correlacionado con la volatilidad
            volumen = 1000 + 5000 * abs(cambios_diarios[i]) * (1 + np.random.normal(0, 0.5))
            
            # Simular algunas anomalías
            if np.random.random() < 0.05:  # 5% de probabilidad de anomalía
                precio *= (1 + np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15))
                volumen *= np.random.uniform(2, 5)
            
            data.append({
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(precio, 2),
                'volume': round(volumen, 2),
                'market_cap': round(precio * volumen, 2)
            })
        
        return data
    
    def extraer_datos_tiempo_real(self, divisa="BTC"):
        """
        Extrae datos en tiempo real para una criptomoneda
        
        Args:
            divisa (str): Símbolo de la criptomoneda
            
        Returns:
            dict: Datos en tiempo real
        """
        self.logger.info(f"Extrayendo datos en tiempo real para {divisa}")
        
        try:
            # Verificar si hay clave API válida
            if self.api_key and self.api_key != "TU_CLAVE_API_AQUI":
                # Intentar Basescan
                datos_basescan = self._extraer_tiempo_real_basescan(divisa)
                
                if datos_basescan:
                    return datos_basescan
                    
                # Si falla, intentar API alternativa
                datos_alternativos = self._extraer_tiempo_real_alternativos(divisa)
                
                if datos_alternativos:
                    return datos_alternativos
                    
                # Si ambos fallan, usar datos ficticios
                return self._generar_datos_tiempo_real(divisa)
            else:
                return self._generar_datos_tiempo_real(divisa)
                
        except Exception as e:
            self.logger.error(f"Error al extraer datos en tiempo real: {str(e)}", exc_info=True)
            return self._generar_datos_tiempo_real(divisa)
    
    def _extraer_tiempo_real_basescan(self, divisa):
        """
        Extrae datos en tiempo real desde Basescan
        
        Args:
            divisa (str): Símbolo de la criptomoneda
            
        Returns:
            dict: Datos en tiempo real o None si falla
        """
        # Endpoint para datos en tiempo real (últimos precios)
        endpoint_price = "/v1/ticker/price"
        params_price = {'symbol': f"{divisa}USDT"}
        
        price_data = self._make_api_request(endpoint_price, params_price)
        
        if not price_data:
            return None
            
        # También consultamos datos de volumen de las últimas 24h
        endpoint_stats = "/v1/ticker/24hr"
        params_stats = {'symbol': f"{divisa}USDT"}
        
        stats_data = self._make_api_request(endpoint_stats, params_stats)
        
        if not stats_data:
            return None
        
        # Combinar datos de ambos endpoints
        return {
            'symbol': divisa,
            'price': float(price_data['price']),
            'volume_24h': float(stats_data['volume']),
            'change_24h': float(stats_data['priceChangePercent']),
            'timestamp': int(datetime.now().timestamp())
        }
    
    def _extraer_tiempo_real_alternativos(self, divisa):
        """
        Extrae datos en tiempo real desde API alternativa
        
        Args:
            divisa (str): Símbolo de la criptomoneda
            
        Returns:
            dict: Datos en tiempo real o None si falla
        """
        try:
            # Ejemplo usando CoinGecko como alternativa
            url = f"https://api.coingecko.com/api/v3/coins/{divisa.lower()}"
            params = {'localization': 'false', 'tickers': 'false', 'community_data': 'false', 'developer_data': 'false'}
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'symbol': divisa,
                'price': data['market_data']['current_price']['usd'],
                'volume_24h': data['market_data']['total_volume']['usd'],
                'change_24h': data['market_data']['price_change_percentage_24h'],
                'timestamp': int(datetime.now().timestamp())
            }
        except Exception as e:
            self.logger.error(f"Error al extraer datos en tiempo real alternativos: {str(e)}")
            return None
    
    def _generar_datos_tiempo_real(self, divisa):
        """
        Genera datos ficticios en tiempo real para desarrollo
        """
        self.logger.warning("Generando datos en tiempo real ficticios")
        import random
        precio_actual = 40000 + random.uniform(-2000, 2000)
        
        return {
            'symbol': divisa,
            'price': precio_actual,
            'volume_24h': 1000000000 + random.uniform(-100000000, 100000000),
            'change_24h': random.uniform(-5, 5),
            'timestamp': int(datetime.now().timestamp())
        }

    def guardar_datos(self, df, ruta_archivo):
        """
        Guarda los datos en un archivo CSV
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            ruta_archivo (str): Ruta donde guardar el archivo
        """
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        try:
            # Guardar el DataFrame
            df.to_csv(ruta_archivo)
            self.logger.info(f"Datos guardados en {ruta_archivo}")
        except Exception as e:
            self.logger.error(f"Error al guardar datos en {ruta_archivo}: {str(e)}")
            raise

    def cargar_datos(self, ruta_archivo):
        """
        Carga datos desde un archivo CSV
        
        Args:
            ruta_archivo (str): Ruta del archivo a cargar
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        if not os.path.exists(ruta_archivo):
            self.logger.error(f"El archivo {ruta_archivo} no existe")
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
        
        try:
            self.logger.info(f"Cargando datos desde {ruta_archivo}")
            df = pd.read_csv(ruta_archivo)
            
            # Si hay una columna fecha, convertirla a datetime y establecerla como índice
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.set_index('fecha')
            elif df.index.name == 'fecha':
                df.index = pd.to_datetime(df.index)
                
            return df
        except Exception as e:
            self.logger.error(f"Error al cargar datos desde {ruta_archivo}: {str(e)}")
            raise
