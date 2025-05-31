"""
Pruebas unitarias para el módulo de extracción de datos
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Importar el módulo a probar
from src.extraccion_datos.extractor import Extractor

class TestExtractor:
    """
    Pruebas para la clase Extractor
    """
    
    @pytest.fixture
    def extractor(self):
        """Fixture que crea una instancia de Extractor para las pruebas"""
        return Extractor("test_key", "https://api.basescan.org", 5)
    
    @pytest.fixture
    def fechas_test(self):
        """Fixture que proporciona fechas de prueba"""
        hoy = datetime.now()
        hace_una_semana = hoy - timedelta(days=7)
        return hace_una_semana, hoy
    
    @pytest.fixture
    def datos_ejemplo(self):
        """Fixture que proporciona datos de ejemplo para las pruebas"""
        hoy = datetime.now()
        datos = []
        
        # Crear 7 días de datos
        for i in range(7):
            fecha = hoy - timedelta(days=i)
            ts = int(fecha.timestamp())
            
            datos.append({
                'timestamp': ts,
                'open': 40000 + i * 100,
                'high': 40100 + i * 100,
                'low': 39900 + i * 100,
                'close': 40050 + i * 100,
                'volume': 1000 + i * 10,
                'market_cap': (40050 + i * 100) * (1000 + i * 10)
            })
            
        return datos
    
    def test_inicializacion(self, extractor):
        """Probar que el extractor se inicializa correctamente"""
        assert extractor.api_key == "test_key"
        assert extractor.api_url == "https://api.basescan.org"
        assert extractor.rate_limit == 5
    
    def test_generar_datos_sinteticos(self, extractor, fechas_test):
        """Probar la generación de datos sintéticos"""
        fecha_inicio, fecha_fin = fechas_test
        datos = extractor._generar_datos_sinteticos(fecha_inicio, fecha_fin)
        
        # Verificar que se generan los datos correctos
        assert len(datos) == 8  # 7 días + 1 (el día final)
        for dato in datos:
            assert all(k in dato for k in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_cap'])
            assert dato['high'] >= dato['low']  # High debe ser mayor o igual que low
    
    @patch('requests.get')
    def test_extraer_datos_historicos_api_valida(self, mock_get, extractor, fechas_test, datos_ejemplo):
        """Probar la extracción de datos históricos con API válida"""
        fecha_inicio, fecha_fin = fechas_test
        
        # Configurar el mock para simular respuesta de API exitosa
        mock_response = MagicMock()
        mock_response.json.return_value = [
            # [timestamp_ms, open, high, low, close, volume, close_time, ...]
            [int(d['timestamp']) * 1000, d['open'], d['high'], d['low'], d['close'], d['volume'], 
             int(d['timestamp']) * 1000 + 999999, 0, 0, 0, 0, 0] 
            for d in datos_ejemplo
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Modificar la clave API para que no sea la predeterminada
        extractor.api_key = "valid_api_key"
        
        # Llamar a la función de extracción
        df = extractor.extraer_datos_historicos(fecha_inicio, fecha_fin, "BTC")
        
        # Verificar que se llamó a la API correctamente
        mock_get.assert_called()
        args, kwargs = mock_get.call_args
        assert kwargs['params']['symbol'] == "BTCUSDT"
        
        # Verificar que el DataFrame tiene los datos correctos
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_extraer_datos_historicos_api_invalida(self, extractor, fechas_test):
        """Probar la extracción de datos históricos con API inválida"""
        fecha_inicio, fecha_fin = fechas_test
        
        # Asegurarse que la clave API es la predeterminada
        extractor.api_key = "TU_CLAVE_API_AQUI"
        
        # Llamar a la función de extracción
        df = extractor.extraer_datos_historicos(fecha_inicio, fecha_fin, "BTC")
        
        # Verificar que se generaron datos sintéticos
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    @patch('requests.get')
    def test_extraer_datos_tiempo_real(self, mock_get, extractor):
        """Probar la extracción de datos en tiempo real"""
        # Configurar mocks para las dos llamadas a la API
        mock_price_response = MagicMock()
        mock_price_response.json.return_value = {"price": "42000.50"}
        mock_price_response.raise_for_status = MagicMock()
        
        mock_stats_response = MagicMock()
        mock_stats_response.json.return_value = {"volume": "12345678.90", "priceChangePercent": "2.5"}
        mock_stats_response.raise_for_status = MagicMock()
        
        # Configurar el mock para que devuelva diferentes respuestas en cada llamada
        mock_get.side_effect = [mock_price_response, mock_stats_response]
        
        # Modificar la clave API para que no sea la predeterminada
        extractor.api_key = "valid_api_key"
        
        # Llamar a la función
        datos = extractor.extraer_datos_tiempo_real("BTC")
        
        # Verificar que se llamó a la API correctamente
        assert mock_get.call_count == 2
        
        # Verificar los datos devueltos
        assert datos['symbol'] == "BTC"
        assert datos['price'] == 42000.50
        assert datos['volume_24h'] == 12345678.90
        assert datos['change_24h'] == 2.5
        assert 'timestamp' in datos
    
    def test_guardar_y_cargar_datos(self, extractor, fechas_test, tmpdir):
        """Probar guardar y cargar datos de archivos"""
        fecha_inicio, fecha_fin = fechas_test
        
        # Generar datos para la prueba
        datos = extractor._generar_datos_sinteticos(fecha_inicio, fecha_fin)
        df = pd.DataFrame(datos)
        
        # Crear un archivo temporal para la prueba
        archivo_test = os.path.join(tmpdir, "test_data.csv")
        
        # Guardar los datos
        extractor.guardar_datos(df, archivo_test)
        
        # Verificar que el archivo existe
        assert os.path.exists(archivo_test)
        
        # Cargar los datos
        df_cargado = extractor.cargar_datos(archivo_test)
        
        # Verificar que los datos son correctos
        assert df_cargado.shape[0] == df.shape[0]
        assert all(col in df_cargado.columns for col in df.columns)
    
    def test_cargar_datos_archivo_no_existente(self, extractor):
        """Probar cargar datos de un archivo que no existe"""
        with pytest.raises(FileNotFoundError):
            extractor.cargar_datos("archivo_inexistente.csv")
