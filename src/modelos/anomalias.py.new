"""
Modulo para deteccion de anomalias en datos de criptomonedas
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

class ModeloAnomalias:
    """
    Clase para la detección de anomalías en datos de criptomonedas utilizando Random Forest
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Inicializa el modelo de deteccion de anomalias
        
        Args:
            n_estimators (int): Numero de arboles en el bosque aleatorio
            max_depth (int): Profundidad maxima de los arboles
            random_state (int): Semilla para reproducibilidad
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        self.logger = logging.getLogger(__name__)
        self.caracteristicas = None
    
    def entrenar(self, df):
        """
        Entrena el modelo de deteccion de anomalias
        
        Args:
            df (pd.DataFrame): DataFrame con datos procesados y columna 'anomalia'
            
        Returns:
            self: El modelo entrenado
        """
        self.logger.info("Entrenando modelo de detección de anomalías")
        
        # Verificar si existe la columna objetivo
        if 'anomalia' not in df.columns:
            self.logger.error("No se encontró la columna 'anomalia' en los datos")
            raise ValueError("No se encontró la columna 'anomalia' en los datos")
        
        # Seleccionar características y objetivo
        X, y = self._preparar_datos(df)
        
        # Entrenar modelo
        try:
            self.modelo.fit(X, y)
            
            # Guardar importancia de características
            self.importancia_caracteristicas = pd.Series(
                self.modelo.feature_importances_,
                index=self.caracteristicas
            ).sort_values(ascending=False)
            
            self.logger.info(f"Modelo entrenado con {len(X)} muestras. Distribución de clases: {pd.Series(y).value_counts().to_dict()}")
            self.logger.info(f"Top 3 características importantes: {self.importancia_caracteristicas[:3].to_dict()}")
        except Exception as e:
            self.logger.error(f"Error al entrenar modelo: {str(e)}", exc_info=True)
            raise
        
        return self
    
    def _preparar_datos(self, df):
        """
        Prepara los datos para entrenamiento o predicción
        
        Args:
            df (pd.DataFrame): DataFrame con datos procesados
            
        Returns:
            tuple: (X, y) para entrenamiento o solo X para predicción
        """
        # Características para el modelo
        self.caracteristicas = [
            'retorno', 'volatilidad', 'rsi', 'macd', 'macd_diff',
            'volumen_rel', 'bb_ancho', 'retorno_log'
        ]
        
        # Verificar qué características están disponibles
        self.caracteristicas = [c for c in self.caracteristicas if c in df.columns]
        
        if not self.caracteristicas:
            self.logger.error("No hay características disponibles para el modelo")
            raise ValueError("No hay características disponibles para el modelo")
        
        # Eliminar filas con NaN
        df_limpio = df.dropna(subset=self.caracteristicas)
        
        # Asegurar que todos los datos son de tipo numérico
        for col in self.caracteristicas:
            try:
                df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error al convertir columna {col} a numérico: {str(e)}")
                
        # Eliminar filas con NaN después de conversión a numérico
        df_limpio = df_limpio.dropna(subset=self.caracteristicas)
        
        # Preparar X
        X = df_limpio[self.caracteristicas].astype(float)
        
        # Preparar y si es necesario
        if 'anomalia' in df_limpio.columns:
            y = df_limpio['anomalia']
            return X, y
        
        return X
    
    def predecir(self, df):
        """
        Predice anomalias para los datos proporcionados
        
        Args:
            df (pd.DataFrame): DataFrame con datos procesados
            
        Returns:
            pd.DataFrame: DataFrame con columnas de prediccion añadidas
        """
        self.logger.info("Prediciendo anomalías")
        
        # Hacer copia para no modificar original
        df_result = df.copy()
        
        try:
            # Preparar datos
            X = self._preparar_datos(df)
            
            if len(X) == 0:
                self.logger.warning("No hay datos válidos para predecir después de limpiar")
                # Crear columnas de predicción vacías
                df_result['prob_anomalia'] = np.nan
                df_result['anomalia_pred'] = np.nan
                df_result['severidad_anomalia'] = np.nan
                return df_result
            
            # Predecir probabilidades
            probs = self.modelo.predict_proba(X)
            predicciones = self.modelo.predict(X)
            
            # Añadir probabilidades y predicciones al DataFrame
            df_result.loc[X.index, 'prob_anomalia'] = probs[:, 1]
            df_result.loc[X.index, 'anomalia_pred'] = predicciones
            
            # Calcular severidad de anomalía (combinación de probabilidad y valores extremos)
            if 'retorno' in df_result.columns:
                # Usar valores absolutos y asegurar que son numéricos
                retornos_abs = pd.to_numeric(df_result['retorno'].abs(), errors='coerce')
                df_result.loc[X.index, 'severidad_anomalia'] = probs[:, 1] * retornos_abs.loc[X.index] / 5
            
            self.logger.info(f"Anomalías detectadas: {predicciones.sum()} de {len(X)} muestras")
        
        except Exception as e:
            self.logger.error(f"Error al predecir anomalías: {str(e)}", exc_info=True)
            # En caso de error, crear columnas de predicción con valores predeterminados
            df_result['prob_anomalia'] = 0
            df_result['anomalia_pred'] = 0
            df_result['severidad_anomalia'] = 0
            
        return df_result
    
    def entrenar_y_predecir(self, df):
        """
        Entrena el modelo y predice anomalías en un solo paso
        
        Args:
            df (pd.DataFrame): DataFrame con datos procesados y columna 'anomalia'
            
        Returns:
            pd.DataFrame: DataFrame con columnas de predicción añadidas
        """
        try:
            self.entrenar(df)
            return self.predecir(df)
        except Exception as e:
            self.logger.error(f"Error en entrenar_y_predecir: {str(e)}", exc_info=True)
            # En caso de error, devolver el DataFrame original con columnas adicionales
            df_result = df.copy()
            df_result['prob_anomalia'] = 0
            df_result['anomalia_pred'] = 0
            df_result['severidad_anomalia'] = 0
            return df_result
    
    def guardar_modelo(self, ruta_archivo):
        """
        Guarda el modelo entrenado a disco
        
        Args:
            ruta_archivo (str): Ruta donde guardar el modelo
        """
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        try:
            # Guardar el modelo y metadatos
            modelo_data = {
                'modelo': self.modelo,
                'caracteristicas': self.caracteristicas,
                'importancia_caracteristicas': getattr(self, 'importancia_caracteristicas', None)
            }
            
            joblib.dump(modelo_data, ruta_archivo)
            self.logger.info(f"Modelo de anomalías guardado en {ruta_archivo}")
        except Exception as e:
            self.logger.error(f"Error al guardar modelo: {str(e)}", exc_info=True)
            raise
    
    def cargar_modelo(self, ruta_archivo):
        """
        Carga un modelo previamente entrenado desde disco
        
        Args:
            ruta_archivo (str): Ruta del modelo a cargar
        """
        if not os.path.exists(ruta_archivo):
            self.logger.error(f"El archivo de modelo {ruta_archivo} no existe")
            raise FileNotFoundError(f"El archivo de modelo {ruta_archivo} no existe")
        
        try:
            self.logger.info(f"Cargando modelo de anomalías desde {ruta_archivo}")
            
            # Cargar modelo y metadatos
            modelo_data = joblib.load(ruta_archivo)
            
            self.modelo = modelo_data['modelo']
            self.caracteristicas = modelo_data['caracteristicas']
            self.importancia_caracteristicas = modelo_data.get('importancia_caracteristicas', None)
        except Exception as e:
            self.logger.error(f"Error al cargar modelo desde {ruta_archivo}: {str(e)}", exc_info=True)
            raise
        
        return self
    
    def analizar_anomalias(self, df):
        """
        Analiza las anomalías detectadas
        
        Args:
            df (pd.DataFrame): DataFrame con predicciones de anomalías
            
        Returns:
            dict: Estadísticas sobre las anomalías detectadas
        """
        self.logger.info("Analizando anomalías detectadas")
        
        if 'anomalia_pred' not in df.columns:
            self.logger.error("No se encontró la columna 'anomalia_pred' en los datos")
            raise ValueError("No se encontró la columna 'anomalia_pred' en los datos")
        
        # Filtrar días con anomalías
        try:
            anomalias = df[df['anomalia_pred'] == 1]
            
            resultados = {
                'total_anomalias': len(anomalias),
                'porcentaje_anomalias': len(anomalias) / len(df) * 100 if len(df) > 0 else 0,
                'anomalias_por_anio': anomalias.groupby(anomalias.index.year).size().to_dict(),
                'retorno_promedio': anomalias['retorno'].mean() if 'retorno' in anomalias.columns else None,
                'volatilidad_promedio': anomalias['volatilidad'].mean() if 'volatilidad' in anomalias.columns else None,
                'top_anomalias': {}
            }
            
            # Top 5 anomalías más severas
            if 'severidad_anomalia' in anomalias.columns:
                top_anomalias = anomalias.sort_values('severidad_anomalia', ascending=False).head(5)
                
                for idx, row in top_anomalias.iterrows():
                    resultados['top_anomalias'][str(idx.date())] = {
                        'retorno': row.get('retorno', None),
                        'volumen_rel': row.get('volumen_rel', None),
                        'severidad': row.get('severidad_anomalia', None)
                    }
            
            return resultados
        except Exception as e:
            self.logger.error(f"Error al analizar anomalías: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'total_anomalias': 0,
                'porcentaje_anomalias': 0
            }
