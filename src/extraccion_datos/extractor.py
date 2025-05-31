"""
Modulo para la extraccion de datos historicos de criptomonedas desde API de Basescan
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

class Extractor:
    """
    Clase para la extraccion de datos historicos de criptomonedas
    """
    
    def __init__(self, api_key, api_url, rate_limit=5):
        """
        Inicializa el extractor de datos
        
        Args:
            api_key (str): Clave de API para Basescan
            api_url (str): URL base de la API
            rate_limit (int): Límite de solicitudes por segundo
        """
        self.api_key = api_key
        self.api_url = api_url
        self.rate_limit = rate_limit
        self.logger = logging.getLogger(__name__)
        
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
            # Usar la API real de Basescan si hay una clave API válida
            if self.api_key and self.api_key != "TU_CLAVE_API_AQUI":
                # Endpoint para datos historicos
                endpoint = f"{self.api_url}/v1/klines"
                
                params = {
                    'apikey': self.api_key,
                    'symbol': f"{divisa}USDT",
                    'interval': '1d',
                    'startTime': inicio_ts * 1000,  # Basescan usa milisegundos
                    'endTime': fin_ts * 1000,
                    'limit': 1000  # Máximo número de registros
                }
                
                self.logger.debug(f"Realizando solicitud a {endpoint}")
                
                data = []
                current_start = inicio_ts * 1000
                
                # Hacer múltiples llamadas si el período es grande (paginación)
                while current_start < fin_ts * 1000:
                    params['startTime'] = current_start
                      # Implementar limitación de tasa
                    time.sleep(1.0 / self.rate_limit)
                    
                    try:
                        response = requests.get(endpoint, params=params, timeout=10)
                        response.raise_for_status()
                        batch_data = response.json()
                        
                        if not batch_data:
                            self.logger.warning(f"No se obtuvieron datos para el periodo {datetime.fromtimestamp(current_start/1000)}")
                            break
                    
                    except requests.exceptions.RequestException as e:
                        self.logger.error(f"Error en la solicitud a la API de Basescan: {str(e)}")
                        # Intentar de nuevo con un retraso
                        time.sleep(5)
                        continue
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Error procesando datos de la API: {str(e)}")
                        break
                    
                    # Convertir datos de la API a nuestro formato
                    for item in batch_data:
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
                    
                    if len(batch_data) < 1000:
                        break
                    
                    # Establecer siguiente punto de inicio
                    current_start = batch_data[-1][0] + 1
                
                self.logger.info(f"Datos extraídos de la API: {len(data)} registros")
            else:
                # Si no hay clave API válida, usar datos sintéticos
                self.logger.warning("Clave API no válida, generando datos sintéticos")
                data = self._generar_datos_sinteticos(fecha_inicio, fecha_fin)
            
            # Crear DataFrame a partir de los datos
            df = pd.DataFrame(data)
            
            # Convertir la columna de timestamp a datetime
            df['fecha'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('fecha')
            
            self.logger.info(f"Datos extraídos con éxito: {len(df)} registros")
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error al extraer datos: {str(e)}")
            raise
    
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
                # Endpoint para datos en tiempo real (últimos precios)
                endpoint = f"{self.api_url}/v1/ticker/price"
                
                params = {
                    'apikey': self.api_key,
                    'symbol': f"{divisa}USDT"
                }
                
                # También consultamos datos de volumen de las últimas 24h
                endpoint_stats = f"{self.api_url}/v1/ticker/24hr"
                
                # Implementar limitación de tasa
                time.sleep(1.0 / self.rate_limit)
                
                try:
                    response = requests.get(endpoint, params=params, timeout=5)
                    response.raise_for_status()
                    price_data = response.json()
                    
                    time.sleep(1.0 / self.rate_limit)
                    
                    response_stats = requests.get(endpoint_stats, params=params, timeout=5)
                    response_stats.raise_for_status()
                    stats_data = response_stats.json()
                    
                    # Combinar datos de ambos endpoints
                    return {
                        'symbol': divisa,
                        'price': float(price_data['price']),
                        'volume_24h': float(stats_data['volume']),
                        'change_24h': float(stats_data['priceChangePercent']),
                        'timestamp': int(datetime.now().timestamp())
                    }
                except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                    self.logger.error(f"Error obteniendo datos en tiempo real: {str(e)}")
                    # En caso de error, generar datos ficticios
                    return self._generar_datos_tiempo_real(divisa)
            else:
                return self._generar_datos_tiempo_real(divisa)
                
        except Exception as e:
            self.logger.error(f"Error al extraer datos en tiempo real: {str(e)}", exc_info=True)
            return self._generar_datos_tiempo_real(divisa)
    
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
