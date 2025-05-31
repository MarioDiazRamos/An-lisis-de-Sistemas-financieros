"""
Utilidades generales para el sistema de mineria de datos
"""

import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Utilidades:
    """
    Clase con funciones utilitarias para el sistema
    """
    
    @staticmethod
    def configurar_logging(log_file, log_level=logging.INFO):
        """
        Configura el sistema de logs
        
        Args:
            log_file (str): Ruta al archivo de log
            log_level (int): Nivel de logging
        """
        # Asegurarse que existe el directorio
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    @staticmethod
    def guardar_json(datos, ruta_archivo):
        """
        Guarda datos en formato JSON
        
        Args:
            datos (dict): Diccionario con datos
            ruta_archivo (str): Ruta donde guardar el archivo
        """
        # Asegurarse que existe el directorio
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Convertir fechas a string para JSON
        def convertir_fechas(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.strftime('%Y-%m-%d')
            raise TypeError(f"Tipo no serializable: {type(obj)}")
        
        # Guardar como JSON
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(datos, f, default=convertir_fechas, indent=2, ensure_ascii=False)
    
    @staticmethod
    def cargar_json(ruta_archivo):
        """
        Carga datos desde un archivo JSON
        
        Args:
            ruta_archivo (str): Ruta del archivo a cargar
            
        Returns:
            dict: Datos cargados
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
        
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def crear_grafico_precios(df, columnas_precio=['close'], titulo='Precio Bitcoin', 
                             guardar_como=None, incluir_vol=True):
        """
        Crea un gráfico de precios
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            columnas_precio (list): Lista de columnas de precio a graficar
            titulo (str): Título del gráfico
            guardar_como (str): Ruta para guardar el gráfico
            incluir_vol (bool): Si se incluye volumen
            
        Returns:
            matplotlib.figure.Figure: Figura creada
        """
        # Crear figura
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Graficar precios
        for col in columnas_precio:
            if col in df.columns:
                ax1.plot(df.index, df[col], label=col)
        
        ax1.set_ylabel('Precio (USD)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # Incluir volumen si se solicita
        if incluir_vol and 'volume' in df.columns:
            ax2 = ax1.twinx()
            ax2.fill_between(df.index, 0, df['volume'], color='lightgray', alpha=0.3, label='Volumen')
            ax2.set_ylabel('Volumen', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            
        # Formatear gráfico
        plt.title(titulo)
        ax1.legend(loc='upper left')
        fig.tight_layout()
        
        # Guardar si se especificó ruta
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def crear_grafico_clusters(df, clusters_col='cluster', precio_col='close', 
                              titulo='Clustering de Bitcoin', guardar_como=None):
        """
        Crea un gráfico de precios coloreados por cluster
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            clusters_col (str): Nombre de la columna con clusters
            precio_col (str): Nombre de la columna con precios
            titulo (str): Título del gráfico
            guardar_como (str): Ruta para guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura creada
        """
        if clusters_col not in df.columns or precio_col not in df.columns:
            raise ValueError(f"Columnas requeridas no encontradas: {clusters_col}, {precio_col}")
        
        # Crear figura
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Colores para clusters
        colores = ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099', '#0099c6']
        
        # Graficar cada cluster por separado
        for cluster in sorted(df[clusters_col].unique()):
            if cluster < 0:  # Saltar puntos no asignados
                continue
            
            mask = df[clusters_col] == cluster
            ax.scatter(
                df.index[mask], 
                df[precio_col][mask], 
                s=30, 
                c=colores[cluster % len(colores)], 
                alpha=0.7, 
                label=f'Cluster {cluster}'
            )
        
        # Añadir línea de precio
        ax.plot(df.index, df[precio_col], color='gray', alpha=0.3, zorder=0)
        
        # Formatear gráfico
        ax.set_title(titulo)
        ax.set_ylabel('Precio (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Guardar si se especificó ruta
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def crear_grafico_anomalias(df, anomalias_col='anomalia_pred', precio_col='close', 
                               titulo='Anomalías Detectadas', guardar_como=None):
        """
        Crea un gráfico destacando días con anomalías
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            anomalias_col (str): Nombre de la columna con anomalías
            precio_col (str): Nombre de la columna con precios
            titulo (str): Título del gráfico
            guardar_como (str): Ruta para guardar el gráfico
            
        Returns:
            matplotlib.figure.Figure: Figura creada
        """
        if anomalias_col not in df.columns or precio_col not in df.columns:
            raise ValueError(f"Columnas requeridas no encontradas: {anomalias_col}, {precio_col}")
        
        # Crear figura
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Graficar precio
        ax.plot(df.index, df[precio_col], color='blue', alpha=0.6, zorder=1)
        
        # Destacar anomalías
        mask_anomalias = df[anomalias_col] == 1
        ax.scatter(
            df.index[mask_anomalias], 
            df[precio_col][mask_anomalias], 
            color='red', 
            s=80, 
            marker='o', 
            label='Anomalía', 
            zorder=2
        )
        
        # Formatear gráfico
        ax.set_title(titulo)
        ax.set_ylabel('Precio (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Guardar si se especificó ruta
        if guardar_como:
            plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def formatear_reglas_para_display(reglas, top_n=10, incluir_cols=None):
        """
        Formatea reglas para mostrarlas de forma legible
        
        Args:
            reglas (pd.DataFrame): DataFrame con reglas
            top_n (int): Número de reglas a incluir
            incluir_cols (list): Columnas a incluir
            
        Returns:
            pd.DataFrame: DataFrame formateado
        """
        if reglas.empty:
            return pd.DataFrame()
        
        # Seleccionar columnas
        if incluir_cols is None:
            incluir_cols = ['antecedentes', 'consecuentes', 'confianza', 'lift']
            
        reglas_disp = reglas[incluir_cols].copy()
        
        # Limitar número de reglas
        if len(reglas_disp) > top_n:
            reglas_disp = reglas_disp.head(top_n)
        
        # Formatear valores numéricos
        for col in ['confianza', 'lift', 'soporte']:
            if col in reglas_disp.columns:
                reglas_disp[col] = reglas_disp[col].map('{:.3f}'.format)
        
        return reglas_disp
