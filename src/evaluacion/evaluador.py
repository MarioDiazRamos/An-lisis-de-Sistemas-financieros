"""
Modulo para evaluacion de modelos de mineria de datos para criptomonedas
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    silhouette_score, confusion_matrix
)

class Evaluador:
    """
    Clase para la evaluacion de modelos de mineria de datos
    """
    
    def __init__(self, parametros):
        """
        Inicializa el evaluador
        
        Args:
            parametros (dict): Diccionario con parametros de configuracion
        """
        self.parametros = parametros
        self.logger = logging.getLogger(__name__)
    
    def evaluar_modelos(self, df_procesado, df_clusters, df_anomalias, reglas):
        """
        Evalua todos los modelos
        
        Args:
            df_procesado (pd.DataFrame): DataFrame con datos procesados
            df_clusters (pd.DataFrame): DataFrame con resultados de clustering
            df_anomalias (pd.DataFrame): DataFrame con resultados de deteccion de anomalias
            reglas (pd.DataFrame): DataFrame con reglas de asociacion
            
        Returns:
            dict: Diccionario con resultados de evaluacion para cada modelo
        """
        self.logger.info("Evaluando todos los modelos")
        
        resultados = {}
        
        # Evaluar clustering
        if 'cluster' in df_clusters.columns:
            resultados['clustering'] = self.evaluar_clustering(df_clusters)
        
        # Evaluar deteccion de anomalias
        if 'anomalia_pred' in df_anomalias.columns and 'anomalia' in df_anomalias.columns:
            resultados['anomalias'] = self.evaluar_anomalias(df_anomalias)
        
        # Evaluar reglas de asociacion
        if not reglas.empty:
            resultados['reglas'] = self.evaluar_reglas(reglas)
        
        # Evaluar rentabilidad de trading
        if 'close' in df_procesado.columns:
            resultados['rentabilidad'] = self.evaluar_rentabilidad(df_procesado, df_anomalias, reglas)
        
        return resultados
    
    def evaluar_clustering(self, df):
        """
        Evalua el modelo de clustering
        
        Args:
            df (pd.DataFrame): DataFrame con resultados de clustering
            
        Returns:
            dict: Metricas de evaluacion
        """
        self.logger.info("Evaluando modelo de clustering")
        
        resultados = {}
        
        # Verificar que tenemos la columna cluster
        if 'cluster' not in df.columns:
            self.logger.error("No se encontró la columna 'cluster' en los datos")
            return {'error': 'No se encontró la columna cluster'}
        
        # Características para calcular silueta
        caracteristicas = [
            'retorno', 'volatilidad', 'rsi', 'macd', 'volumen_rel'
        ]
        
        # Verificar qué características están disponibles
        caracteristicas_disp = [c for c in caracteristicas if c in df.columns]
        
        # Filtrar filas con valores válidos y cluster asignado
        df_val = df[df['cluster'] >= 0].dropna(subset=caracteristicas_disp)
        
        # Calcular coeficiente de silueta si hay suficientes muestras
        if len(df_val) > 1 and len(df_val['cluster'].unique()) > 1:
            try:
                silueta = silhouette_score(
                    df_val[caracteristicas_disp], 
                    df_val['cluster']
                )
                resultados['silhouette_score'] = silueta
            except Exception as e:
                self.logger.error(f"Error al calcular silueta: {str(e)}")
                resultados['silhouette_score'] = None
        
        # Calcular estadísticas por cluster
        cluster_stats = {}
        for cluster in sorted(df['cluster'].unique()):
            if cluster < 0:  # Saltar puntos no asignados
                continue
                
            cluster_data = df[df['cluster'] == cluster]
            
            # Tamaño y porcentaje
            cluster_stats[f'cluster_{cluster}_tamaño'] = len(cluster_data)
            cluster_stats[f'cluster_{cluster}_porcentaje'] = len(cluster_data) / len(df) * 100
            
            # Características promedio
            for caract in caracteristicas_disp:
                cluster_stats[f'cluster_{cluster}_{caract}_promedio'] = cluster_data[caract].mean()
            
            # Anomalías en el cluster
            if 'anomalia' in df.columns:
                cluster_stats[f'cluster_{cluster}_porc_anomalias'] = (cluster_data['anomalia'] > 0).mean() * 100
        
        resultados.update(cluster_stats)
        
        # Calcular inercia (suma de distancias al cuadrado a los centros)
        # Esto requeriría acceso al objeto KMeans directamente, así que lo omitimos aquí
        
        return resultados
    
    def evaluar_anomalias(self, df):
        """
        Evalua el modelo de deteccion de anomalias
        
        Args:
            df (pd.DataFrame): DataFrame con resultados de deteccion de anomalias
            
        Returns:
            dict: Metricas de evaluacion
        """
        self.logger.info("Evaluando modelo de detección de anomalías")
        
        resultados = {}
        
        # Verificar que tenemos las columnas necesarias
        if 'anomalia' not in df.columns or 'anomalia_pred' not in df.columns:
            self.logger.error("No se encontraron las columnas 'anomalia' o 'anomalia_pred'")
            return {'error': 'Columnas requeridas no encontradas'}
        
        # Filtrar filas con valores válidos
        df_val = df.dropna(subset=['anomalia', 'anomalia_pred'])
        
        # Convertir a enteros (asegurar que son 0 y 1)
        y_true = df_val['anomalia'].astype(int)
        y_pred = df_val['anomalia_pred'].astype(int)
        
        # Calcular métricas
        resultados['precision'] = precision_score(y_true, y_pred, zero_division=0)
        resultados['recall'] = recall_score(y_true, y_pred, zero_division=0)
        resultados['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            resultados['tn'], resultados['fp'], resultados['fn'], resultados['tp'] = cm.ravel()
        
        # Porcentaje de anomalías detectadas
        num_anomalias_reales = y_true.sum()
        num_anomalias_detectadas = y_pred.sum()
        
        resultados['num_anomalias_reales'] = int(num_anomalias_reales)
        resultados['num_anomalias_detectadas'] = int(num_anomalias_detectadas)
        resultados['porc_anomalias_reales'] = num_anomalias_reales / len(y_true) * 100
        resultados['porc_anomalias_detectadas'] = num_anomalias_detectadas / len(y_pred) * 100
        
        return resultados
    
    def evaluar_reglas(self, reglas):
        """
        Evalua las reglas de asociacion
        
        Args:
            reglas (pd.DataFrame): DataFrame con reglas de asociacion
            
        Returns:
            dict: Metricas de evaluacion
        """
        self.logger.info("Evaluando reglas de asociación")
        
        resultados = {}
        
        # Verificar que hay reglas
        if reglas.empty:
            self.logger.error("No hay reglas para evaluar")
            return {'error': 'No hay reglas para evaluar'}
        
        # Estadísticas generales
        resultados['num_reglas'] = len(reglas)
        resultados['confianza_promedio'] = reglas['confianza'].mean()
        resultados['lift_promedio'] = reglas['lift'].mean()
        resultados['soporte_promedio'] = reglas['soporte'].mean()
        
        # Estadísticas para reglas predictivas
        reglas_pred = reglas[reglas['consecuentes'].str.contains('proximo_retorno')]
        
        if not reglas_pred.empty:
            resultados['num_reglas_pred'] = len(reglas_pred)
            resultados['confianza_promedio_pred'] = reglas_pred['confianza'].mean()
            resultados['lift_promedio_pred'] = reglas_pred['lift'].mean()
            
            # Número de reglas para "sube", "baja", "neutral"
            for resultado in ['sube', 'baja', 'neutral']:
                mask = reglas_pred['consecuentes'].str.contains(resultado)
                resultados[f'num_reglas_{resultado}'] = mask.sum()
                if mask.sum() > 0:
                    resultados[f'confianza_promedio_{resultado}'] = reglas_pred[mask]['confianza'].mean()
                    resultados[f'lift_promedio_{resultado}'] = reglas_pred[mask]['lift'].mean()
        
        return resultados
    
    def evaluar_rentabilidad(self, df_procesado, df_anomalias, reglas, periodo_test=None):
        """
        Evalua la rentabilidad de diferentes estrategias de trading
        
        Args:
            df_procesado (pd.DataFrame): DataFrame con datos procesados
            df_anomalias (pd.DataFrame): DataFrame con resultados de anomalías
            reglas (pd.DataFrame): DataFrame con reglas de asociación
            periodo_test (tuple): Tupla con (fecha_inicio, fecha_fin) para test
            
        Returns:
            dict: Métricas de rentabilidad
        """
        self.logger.info("Evaluando rentabilidad de estrategias de trading")
        
        resultados = {}
        
        # Verificar columnas necesarias
        if 'close' not in df_procesado.columns:
            self.logger.error("No se encontró la columna 'close' para evaluar rentabilidad")
            return {'error': 'Columna close no encontrada'}
        
        # Seleccionar período de prueba (por defecto, último 20%)
        if periodo_test is None:
            n_test = int(len(df_procesado) * 0.2)
            df_test = df_procesado.iloc[-n_test:]
        else:
            inicio, fin = periodo_test
            df_test = df_procesado.loc[inicio:fin]
        
        # Asegurarse que están en el mismo índice
        df_anomalias = df_anomalias.loc[df_test.index] if df_test.index.isin(df_anomalias.index).any() else df_anomalias
        
        # Estrategia 1: Buy and Hold
        rentabilidad_bh = self._calcular_buy_hold(df_test)
        resultados['rentabilidad_buy_hold'] = rentabilidad_bh
        
        # Estrategia 2: Media Móvil Simple (benchmark)
        rentabilidad_sma, trades_sma = self._calcular_estrategia_sma(df_test)
        resultados['rentabilidad_sma'] = rentabilidad_sma
        resultados['trades_sma'] = trades_sma
        
        # Estrategia 3: Basada en anomalías
        if 'anomalia_pred' in df_anomalias.columns:
            rentabilidad_anom, trades_anom = self._calcular_estrategia_anomalias(df_test, df_anomalias)
            resultados['rentabilidad_anomalias'] = rentabilidad_anom
            resultados['trades_anomalias'] = trades_anom
        
        # Estrategia 4: Basada en reglas de asociación
        if not reglas.empty:
            # Esta función requeriría implementar la aplicación de reglas para cada día
            # lo cual es más complejo y dependería de cómo estructuramos las reglas
            # Lo dejamos como ejercicio futuro
            pass
        
        return resultados
    
    def _calcular_buy_hold(self, df):
        """
        Calcula la rentabilidad de buy and hold
        
        Args:
            df (pd.DataFrame): DataFrame con precios
            
        Returns:
            float: Rentabilidad porcentual
        """
        precio_inicio = df['close'].iloc[0]
        precio_fin = df['close'].iloc[-1]
        
        return (precio_fin / precio_inicio - 1) * 100
    
    def _calcular_estrategia_sma(self, df, ventana_corta=20, ventana_larga=50):
        """
        Calcula la rentabilidad de una estrategia basada en medias móviles
        
        Args:
            df (pd.DataFrame): DataFrame con precios
            ventana_corta (int): Ventana para la media móvil corta
            ventana_larga (int): Ventana para la media móvil larga
            
        Returns:
            tuple: (rentabilidad, número de operaciones)
        """
        # Calcular medias móviles si no existen
        if f'sma_{ventana_corta}' not in df.columns:
            df[f'sma_{ventana_corta}'] = df['close'].rolling(window=ventana_corta).mean()
        
        if f'sma_{ventana_larga}' not in df.columns:
            df[f'sma_{ventana_larga}'] = df['close'].rolling(window=ventana_larga).mean()
        
        # Crear señales: 1 = comprar, -1 = vender, 0 = mantener
        df['señal'] = 0
        df.loc[df[f'sma_{ventana_corta}'] > df[f'sma_{ventana_larga}'], 'señal'] = 1
        df.loc[df[f'sma_{ventana_corta}'] <= df[f'sma_{ventana_larga}'], 'señal'] = -1
        
        # Detectar cambios en la señal (cruces)
        df['cambio_señal'] = df['señal'].diff().fillna(0)
        
        # Simular operaciones
        capital = self.parametros['capital_inicial']
        comision = self.parametros['comision'] / 100
        
        posicion = 0  # 0 = sin posición, 1 = comprado
        precio_compra = 0
        operaciones = 0
        
        for idx, row in df.iterrows():
            if posicion == 0 and row['cambio_señal'] > 0:
                # Comprar
                precio_compra = row['close']
                posicion = 1
                capital -= capital * comision  # Comisión de compra
                operaciones += 1
                
            elif posicion == 1 and row['cambio_señal'] < 0:
                # Vender
                precio_venta = row['close']
                retorno = precio_venta / precio_compra - 1
                capital *= (1 + retorno)
                capital -= capital * comision  # Comisión de venta
                posicion = 0
                operaciones += 1
        
        # Si terminamos con posición abierta, cerrar al último precio
        if posicion == 1:
            precio_venta = df['close'].iloc[-1]
            retorno = precio_venta / precio_compra - 1
            capital *= (1 + retorno)
            capital -= capital * comision
        
        # Calcular rentabilidad respecto al capital inicial
        rentabilidad = (capital / self.parametros['capital_inicial'] - 1) * 100
        
        return rentabilidad, operaciones
    
    def _calcular_estrategia_anomalias(self, df, df_anomalias, umbral_prob=0.7):
        """
        Calcula la rentabilidad de una estrategia basada en anomalías
        
        Args:
            df (pd.DataFrame): DataFrame con precios
            df_anomalias (pd.DataFrame): DataFrame con probabilidades de anomalías
            umbral_prob (float): Umbral de probabilidad para considerar anomalía
            
        Returns:
            tuple: (rentabilidad, número de operaciones)
        """
        # Asegurarse que están en el mismo índice
        df_comun = df.merge(
            df_anomalias[['anomalia_pred', 'prob_anomalia']], 
            left_index=True, 
            right_index=True,
            how='left'
        )
        
        # Estrategia: Comprar cuando se detecta anomalía, vender después de N días
        dias_mantenimiento = 5
        
        # Inicializar variables
        capital = self.parametros['capital_inicial']
        comision = self.parametros['comision'] / 100
        
        posicion = 0  # 0 = sin posición, 1 = comprado
        precio_compra = 0
        dias_desde_compra = 0
        operaciones = 0
        
        for idx, row in df_comun.iterrows():
            if posicion == 0 and row.get('prob_anomalia', 0) > umbral_prob:
                # Comprar en anomalía
                precio_compra = row['close']
                posicion = 1
                capital -= capital * comision  # Comisión de compra
                dias_desde_compra = 0
                operaciones += 1
                
            elif posicion == 1:
                # Incrementar contador
                dias_desde_compra += 1
                
                # Vender después de N días
                if dias_desde_compra >= dias_mantenimiento:
                    precio_venta = row['close']
                    retorno = precio_venta / precio_compra - 1
                    capital *= (1 + retorno)
                    capital -= capital * comision  # Comisión de venta
                    posicion = 0
                    operaciones += 1
        
        # Si terminamos con posición abierta, cerrar al último precio
        if posicion == 1:
            precio_venta = df_comun['close'].iloc[-1]
            retorno = precio_venta / precio_compra - 1
            capital *= (1 + retorno)
            capital -= capital * comision
        
        # Calcular rentabilidad respecto al capital inicial
        rentabilidad = (capital / self.parametros['capital_inicial'] - 1) * 100
        
        return rentabilidad, operaciones
    
    def generar_reporte(self, resultados):
        """
        Genera un reporte en formato de diccionario con los resultados de evaluación
        
        Args:
            resultados (dict): Diccionario con los resultados de evaluación
            
        Returns:
            dict: Reporte con los resultados más importantes
        """
        self.logger.info("Generando reporte de resultados")
        
        reporte = {}
        
        # Incluir métricas de clustering
        if 'clustering' in resultados:
            reporte['clustering'] = {
                'calidad_clusters': resultados['clustering'].get('silhouette_score', 0),
                'num_clusters': len([k for k in resultados['clustering'].keys() if k.startswith('cluster_')]),
                'distribucion': {}
            }
            
            # Extraer información de cada cluster
            for k, v in resultados['clustering'].items():
                if k.startswith('cluster_') and k.endswith('_tamaño'):
                    cluster_id = k.split('_')[1]
                    reporte['clustering']['distribucion'][f'Cluster {cluster_id}'] = {
                        'tamaño': resultados['clustering'].get(f'cluster_{cluster_id}_tamaño', 0),
                        'porcentaje': resultados['clustering'].get(f'cluster_{cluster_id}_porcentaje', 0),
                        'retorno_promedio': resultados['clustering'].get(f'cluster_{cluster_id}_retorno_promedio', 0)
                    }
        
        # Incluir métricas de anomalías
        if 'anomalias' in resultados:
            reporte['anomalias'] = {
                'precision': resultados['anomalias'].get('precision', 0),
                'recall': resultados['anomalias'].get('recall', 0),
                'f1': resultados['anomalias'].get('f1_score', 0),
                'num_anomalias': resultados['anomalias'].get('num_anomalias_detectadas', 0),
                'porcentaje': resultados['anomalias'].get('porc_anomalias_detectadas', 0)
            }
        
        # Incluir métricas de reglas
        if 'reglas' in resultados:
            reporte['reglas'] = {
                'total_reglas': resultados['reglas'].get('num_reglas', 0),
                'confianza_promedio': resultados['reglas'].get('confianza_promedio', 0),
                'lift_promedio': resultados['reglas'].get('lift_promedio', 0),
                'reglas_predictivas': resultados['reglas'].get('num_reglas_pred', 0)
            }
        
        # Incluir métricas de rentabilidad
        if 'rentabilidad' in resultados:
            reporte['rentabilidad'] = {
                'buy_hold': resultados['rentabilidad'].get('rentabilidad_buy_hold', 0),
                'estrategia_sma': resultados['rentabilidad'].get('rentabilidad_sma', 0),
                'estrategia_anomalias': resultados['rentabilidad'].get('rentabilidad_anomalias', 0),
                'trades_realizados': resultados['rentabilidad'].get('trades_sma', 0) + 
                                    resultados['rentabilidad'].get('trades_anomalias', 0)
            }
        
        # Convertir tipos numpy a tipos nativos de Python para correcta serialización JSON
        from src.utils.json_utils import convert_numpy_types
        reporte = convert_numpy_types(reporte)
        
        return reporte
