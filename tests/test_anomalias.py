"""
Pruebas unitarias para el módulo de detección de anomalías
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Importar el módulo a probar
from src.modelos.anomalias import ModeloAnomalias

class TestModeloAnomalias:
    """
    Pruebas para la clase ModeloAnomalias
    """
    
    @pytest.fixture
    def modelo(self):
        """Fixture que crea una instancia del modelo para las pruebas"""
        return ModeloAnomalias(n_estimators=10, max_depth=3, random_state=42)
    
    @pytest.fixture
    def datos_ejemplo(self):
        """Fixture que proporciona datos de ejemplo para las pruebas"""
        # Crear un DataFrame con 100 filas de datos
        np.random.seed(42)
        
        # Fechas para el índice
        fechas = [datetime.now() - timedelta(days=i) for i in range(100)]
        
        # Crear datos normales
        retorno = np.random.normal(0.001, 0.02, 100)
        volatilidad = np.abs(retorno) * 2
        rsi = np.random.uniform(30, 70, 100)
        macd = np.random.normal(0, 0.5, 100)
        macd_diff = np.random.normal(0, 0.2, 100)
        volumen_rel = np.random.gamma(2, 1, 100)
        bb_ancho = np.random.uniform(0.5, 2, 100)
        retorno_log = np.log1p(retorno)
        
        # Insertar algunas anomalías (5%)
        anomalias_indices = np.random.choice(100, 5, replace=False)
        for idx in anomalias_indices:
            retorno[idx] *= 10  # Anomalía en el retorno
            volumen_rel[idx] *= 5  # Anomalía en el volumen
        
        # Crear DataFrame
        df = pd.DataFrame({
            'retorno': retorno,
            'volatilidad': volatilidad,
            'rsi': rsi,
            'macd': macd,
            'macd_diff': macd_diff,
            'volumen_rel': volumen_rel,
            'bb_ancho': bb_ancho,
            'retorno_log': retorno_log,
            # Etiquetar anomalías para entrenamiento
            'anomalia': [1 if i in anomalias_indices else 0 for i in range(100)]
        }, index=fechas)
        
        return df
    
    def test_inicializacion(self, modelo):
        """Probar que el modelo se inicializa correctamente"""
        assert modelo.n_estimators == 10
        assert modelo.max_depth == 3
        assert modelo.random_state == 42
    
    def test_preparar_datos(self, modelo, datos_ejemplo):
        """Probar la preparación de datos"""
        X, y = modelo._preparar_datos(datos_ejemplo)
        
        # Verificar que los datos se preparan correctamente
        assert X.shape == (100, 8)  # 8 características
        assert y.shape == (100,)  # 100 etiquetas
        
        # Verificar que las columnas son correctas
        assert set(modelo.caracteristicas) == {'retorno', 'volatilidad', 'rsi', 'macd', 'macd_diff',
                                              'volumen_rel', 'bb_ancho', 'retorno_log'}
    
    def test_entrenar(self, modelo, datos_ejemplo):
        """Probar el entrenamiento del modelo"""
        # Entrenar modelo
        modelo.entrenar(datos_ejemplo)
        
        # Verificar que el modelo se entrenó correctamente
        assert hasattr(modelo, 'modelo')
        assert hasattr(modelo, 'importancia_caracteristicas')
        assert len(modelo.importancia_caracteristicas) == 8  # 8 características
    
    def test_predecir(self, modelo, datos_ejemplo):
        """Probar las predicciones del modelo"""
        # Entrenar primero
        modelo.entrenar(datos_ejemplo)
        
        # Predecir
        df_pred = modelo.predecir(datos_ejemplo)
        
        # Verificar resultados
        assert 'prob_anomalia' in df_pred.columns
        assert 'anomalia_pred' in df_pred.columns
        assert 'severidad_anomalia' in df_pred.columns
        assert df_pred['anomalia_pred'].sum() > 0  # Debería haber al menos una anomalía detectada
    
    def test_guardar_cargar_modelo(self, modelo, datos_ejemplo, tmpdir):
        """Probar guardar y cargar el modelo"""
        # Entrenar el modelo
        modelo.entrenar(datos_ejemplo)
        
        # Crear un archivo temporal para guardar el modelo
        archivo_modelo = os.path.join(tmpdir, "modelo_anomalias.pkl")
        
        # Guardar el modelo
        modelo.guardar_modelo(archivo_modelo)
        
        # Verificar que el archivo existe
        assert os.path.exists(archivo_modelo)
        
        # Crear una nueva instancia del modelo y cargar el modelo guardado
        nuevo_modelo = ModeloAnomalias()
        nuevo_modelo.cargar_modelo(archivo_modelo)
        
        # Verificar que los atributos importantes se cargaron correctamente
        assert hasattr(nuevo_modelo, 'modelo')
        assert hasattr(nuevo_modelo, 'caracteristicas')
        assert len(nuevo_modelo.caracteristicas) == 8
        
        # Hacer una predicción para asegurarse de que el modelo funciona
        df_pred = nuevo_modelo.predecir(datos_ejemplo)
        assert 'prob_anomalia' in df_pred.columns
    
    def test_analizar_anomalias(self, modelo, datos_ejemplo):
        """Probar el análisis de anomalías"""
        # Entrenar y predecir
        modelo.entrenar(datos_ejemplo)
        df_pred = modelo.predecir(datos_ejemplo)
        
        # Analizar anomalías
        resultados = modelo.analizar_anomalias(df_pred)
        
        # Verificar resultados
        assert 'total_anomalias' in resultados
        assert 'porcentaje_anomalias' in resultados
        assert 'anomalias_por_anio' in resultados
        assert 'retorno_promedio' in resultados
        assert 'top_anomalias' in resultados
        
        # Debe haber al menos una anomalía en el top
        assert len(resultados['top_anomalias']) > 0
    
    def test_entrenar_sin_columna_anomalia(self, modelo):
        """Probar que el entrenamiento falla si no hay columna 'anomalia'"""
        # Crear datos sin la columna 'anomalia'
        df = pd.DataFrame({
            'retorno': [0.01, 0.02, -0.01],
            'volatilidad': [0.02, 0.03, 0.02]
        })
        
        # El entrenamiento debe fallar
        with pytest.raises(ValueError):
            modelo.entrenar(df)
    
    def test_preparar_datos_sin_caracteristicas(self, modelo):
        """Probar que la preparación de datos falla si no hay características"""
        # Crear datos sin características válidas
        df = pd.DataFrame({
            'otra_columna': [1, 2, 3],
            'anomalia': [0, 0, 1]
        })
        
        # La preparación debe fallar
        with pytest.raises(ValueError):
            modelo._preparar_datos(df)
