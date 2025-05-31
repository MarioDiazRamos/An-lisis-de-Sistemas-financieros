"""
Pruebas unitarias para el módulo de preprocesamiento
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Agregar el directorio raíz al path para importar los módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocesamiento.preprocesador import Preprocesador
import config

class TestPreprocesador(unittest.TestCase):
    """Clase para pruebas del preprocesador"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        # Crear un dataframe de prueba
        fechas = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]
        precios = np.linspace(40000, 45000, 50) + np.random.normal(0, 500, 50)  # Precios con tendencia y ruido
        volumenes = np.random.uniform(1000, 5000, 50)  # Volúmenes aleatorios
        
        self.df_test = pd.DataFrame({
            'fecha': fechas,
            'open': precios - np.random.uniform(0, 100, 50),
            'high': precios + np.random.uniform(0, 200, 50),
            'low': precios - np.random.uniform(0, 200, 50),
            'close': precios,
            'volume': volumenes
        })
        
        # Establecer fecha como índice
        self.df_test['fecha'] = pd.to_datetime(self.df_test['fecha'])
        self.df_test = self.df_test.set_index('fecha')
        
        # Crear instancia del preprocesador
        self.preprocesador = Preprocesador(config.PARAMETROS)
    
    def test_limpiar_datos(self):
        """Prueba la limpieza de datos"""
        # Añadir algunos valores nulos
        df_con_nulos = self.df_test.copy()
        df_con_nulos.loc[df_con_nulos.index[5], 'close'] = np.nan
        df_con_nulos.loc[df_con_nulos.index[10], 'volume'] = np.nan
        
        # Limpiar datos
        df_limpio = self.preprocesador._limpiar_datos(df_con_nulos)
        
        # Verificar que no hay valores nulos
        self.assertEqual(df_limpio.isnull().sum().sum(), 0)
        
        # Verificar que el tamaño es el mismo
        self.assertEqual(len(df_limpio), len(df_con_nulos))
    
    def test_calcular_retornos(self):
        """Prueba el cálculo de retornos"""
        # Calcular retornos
        df_con_retornos = self.preprocesador._calcular_retornos(self.df_test)
        
        # Verificar que se agregaron las columnas de retornos
        self.assertIn('retorno', df_con_retornos.columns)
        self.assertIn('retorno_log', df_con_retornos.columns)
        self.assertIn('retorno_acum', df_con_retornos.columns)
        
        # Verificar cálculo de retornos diarios
        retornos_correctos = self.df_test['close'].pct_change() * 100
        np.testing.assert_array_almost_equal(df_con_retornos['retorno'].values[1:], retornos_correctos.values[1:])
    
    def test_calcular_indicadores(self):
        """Prueba el cálculo de indicadores técnicos"""
        # Agregar retornos primero
        df_con_retornos = self.preprocesador._calcular_retornos(self.df_test)
        
        # Calcular indicadores
        df_con_indicadores = self.preprocesador._calcular_indicadores(df_con_retornos)
        
        # Verificar que se agregaron los indicadores
        for indicador in ['rsi', 'macd', 'volatilidad', 'sma_20', 'ema_20', 'bb_alto', 'bb_bajo']:
            self.assertIn(indicador, df_con_indicadores.columns)
        
        # Verificar que los valores de RSI están en el rango [0, 100]
        self.assertTrue((df_con_indicadores['rsi'] >= 0).all() and (df_con_indicadores['rsi'] <= 100).all())
    
    def test_discretizar(self):
        """Prueba la discretización de variables"""
        # Procesar datos completos
        df_procesado = self.preprocesador.procesar(self.df_test)
        
        # Discretizar
        df_discretizado = self.preprocesador.discretizar(df_procesado)
        
        # Verificar columnas categóricas
        for col in ['rsi_cat', 'retorno_cat', 'volatilidad_cat', 'proximo_retorno_cat']:
            self.assertIn(col, df_discretizado.columns)
            self.assertEqual(df_discretizado[col].dtype.name, 'category')
        
        # Verificar que las etiquetas son correctas para RSI
        rsi_labels = df_discretizado['rsi_cat'].cat.categories.tolist()
        self.assertTrue(all(label in rsi_labels for label in ['bajo', 'medio', 'alto']))

if __name__ == '__main__':
    unittest.main()
