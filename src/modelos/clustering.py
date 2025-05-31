"""
Modulo para clustering de datos de criptomonedas
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

class ModeloClustering:
    """
    Clase para clustering de datos de criptomonedas
    """
    
    def __init__(self, n_clusters=4, random_state=42):
        """
        Inicializa el modelo de clustering
        
        Args:
            n_clusters (int): Numero de clusters a crear
            random_state (int): Semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.modelo = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.logger = logging.getLogger(__name__)
        
    def entrenar(self, df):
        """
        Entrena el modelo de clustering con los datos proporcionados
        
        Args:
            df (pd.DataFrame): DataFrame con datos normalizados
            
        Returns:
            self: El modelo entrenado
        """
        self.logger.info(f"Entrenando modelo de clustering con {self.n_clusters} clusters")
        
        # Seleccionar características para clustering
        X = self._seleccionar_caracteristicas(df)
        
        # Entrenar modelo
        self.modelo.fit(X)
        
        self.logger.info(f"Modelo de clustering entrenado. Centros: {self.modelo.cluster_centers_.shape}")
        return self
    
    def _seleccionar_caracteristicas(self, df):
        """
        Selecciona las caracteristicas relevantes para clustering
        
        Args:
            df (pd.DataFrame): DataFrame con todas las características
            
        Returns:
            np.array: Array con características seleccionadas
        """
        # Características que usaremos para clustering
        caracteristicas = [
            'retorno', 'volatilidad', 'rsi', 'macd', 'volumen_rel'
        ]
        
        # Verificar qué características están disponibles
        caracteristicas_disp = [c for c in caracteristicas if c in df.columns]
        
        if not caracteristicas_disp:
            self.logger.error("No hay características disponibles para clustering")
            raise ValueError("No hay características disponibles para clustering")
        
        self.logger.debug(f"Características seleccionadas para clustering: {caracteristicas_disp}")
        
        # Eliminar filas con NaN
        X = df[caracteristicas_disp].dropna()
        
        return X
    
    def predecir(self, df):
        """
        Predice clusters para los datos proporcionados
        
        Args:
            df (pd.DataFrame): DataFrame con datos normalizados
            
        Returns:
            pd.DataFrame: DataFrame original con columna de cluster añadida
        """
        self.logger.info("Prediciendo clusters")
        
        # Hacer copia para no modificar original
        df_result = df.copy()
        
        # Seleccionar características
        X = self._seleccionar_caracteristicas(df)
        
        # Predecir clusters
        clusters = self.modelo.predict(X)
        
        # Añadir resultados al DataFrame
        df_result.loc[X.index, 'cluster'] = clusters.astype(int)
        
        # Rellenar valores NaN con -1 (para indicar que no se pudo calcular)
        if 'cluster' in df_result.columns:
            df_result['cluster'] = df_result['cluster'].fillna(-1).astype(int)
        
        return df_result
    
    def entrenar_y_predecir(self, df):
        """
        Entrena el modelo y predice los clusters en un solo paso
        
        Args:
            df (pd.DataFrame): DataFrame con datos normalizados
            
        Returns:
            pd.DataFrame: DataFrame con columna de cluster añadida
        """
        self.entrenar(df)
        return self.predecir(df)
    
    def guardar_modelo(self, ruta_archivo):
        """
        Guarda el modelo entrenado a disco
        
        Args:
            ruta_archivo (str): Ruta donde guardar el modelo
        """
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Guardar el modelo
        joblib.dump(self.modelo, ruta_archivo)
        self.logger.info(f"Modelo de clustering guardado en {ruta_archivo}")
    
    def cargar_modelo(self, ruta_archivo):
        """
        Carga un modelo previamente entrenado desde disco
        
        Args:
            ruta_archivo (str): Ruta del modelo a cargar
        """
        if not os.path.exists(ruta_archivo):
            self.logger.error(f"El archivo de modelo {ruta_archivo} no existe")
            raise FileNotFoundError(f"El archivo de modelo {ruta_archivo} no existe")
        
        self.logger.info(f"Cargando modelo de clustering desde {ruta_archivo}")
        self.modelo = joblib.load(ruta_archivo)
        self.n_clusters = self.modelo.n_clusters
        
        return self
    
    def analizar_clusters(self, df):
        """
        Analiza las características de cada cluster
        
        Args:
            df (pd.DataFrame): DataFrame con datos y columna 'cluster'
            
        Returns:
            dict: Diccionario con estadísticas de cada cluster
        """
        self.logger.info("Analizando características de los clusters")
        
        if 'cluster' not in df.columns:
            self.logger.error("No se encontró la columna 'cluster' en los datos")
            raise ValueError("No se encontró la columna 'cluster' en los datos")
        
        # Variables a analizar para cada cluster
        variables = [
            'retorno', 'volatilidad', 'rsi', 'volumen_rel', 
            'macd', 'close', 'anomalia'
        ]
        
        # Verificar qué variables están disponibles
        variables_disp = [v for v in variables if v in df.columns]
        
        resultados = {}
        
        # Para cada cluster, calcular estadísticas
        for cluster in sorted(df['cluster'].unique()):
            if cluster == -1:  # Saltar puntos no asignados
                continue
                
            # Filtrar datos del cluster
            cluster_data = df[df['cluster'] == cluster]
            
            # Calcular estadísticas
            stats = {}
            for var in variables_disp:
                stats[var] = {
                    'media': cluster_data[var].mean(),
                    'mediana': cluster_data[var].median(),
                    'std': cluster_data[var].std(),
                    'min': cluster_data[var].min(),
                    'max': cluster_data[var].max()
                }
            
            # Porcentaje de anomalías en el cluster
            if 'anomalia' in variables_disp:
                stats['porcentaje_anomalias'] = (cluster_data['anomalia'] > 0).mean() * 100
            
            # Fechas representativas del cluster
            stats['fechas_ejemplo'] = cluster_data.index[:5].tolist()
            
            # Tamaño del cluster
            stats['tamaño'] = len(cluster_data)
            stats['porcentaje'] = len(cluster_data) / len(df) * 100
            
            resultados[f'Cluster {cluster}'] = stats
        
        return resultados
