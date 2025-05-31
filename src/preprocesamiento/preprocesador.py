"""
Modulo para el preprocesamiento de datos de criptomonedas
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocesador:
    """
    Clase para el preprocesamiento de datos de criptomonedas
    """
    
    def __init__(self, parametros):
        """
        Inicializa el preprocesador de datos
        
        Args:
            parametros (dict): Diccionario con parametros de configuracion
        """
        self.parametros = parametros
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def procesar(self, df):
        """
        Procesa los datos crudos: limpia, calcula indicadores y prepara para modelado
        
        Args:
            df (pd.DataFrame): DataFrame con datos crudos
            
        Returns:
            pd.DataFrame: DataFrame con datos procesados
        """
        self.logger.info("Iniciando preprocesamiento de datos")
        
        # Hacer una copia para no modificar el original
        df_proc = df.copy()
        
        # Asegurarse que el indice es de tipo datetime
        if not isinstance(df_proc.index, pd.DatetimeIndex):
            if 'fecha' in df_proc.columns:
                df_proc['fecha'] = pd.to_datetime(df_proc['fecha'])
                df_proc = df_proc.set_index('fecha')
            else:
                df_proc.index = pd.to_datetime(df_proc.index)
        
        # 1. Limpieza de datos
        df_proc = self._limpiar_datos(df_proc)
        
        # 2. Calcular retornos
        df_proc = self._calcular_retornos(df_proc)
        
        # 3. Calcular indicadores tecnicos
        df_proc = self._calcular_indicadores(df_proc)
        
        # 4. Normalizar caracteristicas para modelos
        df_proc = self._normalizar_caracteristicas(df_proc)
        
        # 5. Deteccion preliminar de anomalias basada en umbrales
        df_proc = self._detectar_anomalias_umbral(df_proc)
        
        self.logger.info(f"Preprocesamiento completado: {df_proc.shape[0]} filas, {df_proc.shape[1]} columnas")
        return df_proc
    
    def _limpiar_datos(self, df):
        """
        Limpia los datos: elimina filas duplicadas, maneja valores nulos
        
        Args:
            df (pd.DataFrame): DataFrame a limpiar
            
        Returns:
            pd.DataFrame: DataFrame limpio
        """
        self.logger.debug("Limpiando datos...")
        
        # Eliminar filas duplicadas
        df_limpio = df.drop_duplicates()
        
        # Rellenar valores nulos
        if df_limpio.isnull().any().any():
            self.logger.warning(f"Se encontraron {df_limpio.isnull().sum().sum()} valores nulos")
            
            # Para precios, usar interpolación lineal
            for col in ['open', 'high', 'low', 'close']:
                if col in df_limpio.columns and df_limpio[col].isnull().any():
                    df_limpio[col] = df_limpio[col].interpolate(method='linear')
            
            # Para volumen, usar la media
            if 'volume' in df_limpio.columns and df_limpio['volume'].isnull().any():
                df_limpio['volume'] = df_limpio['volume'].fillna(df_limpio['volume'].mean())
        
        # Ordenar por fecha
        df_limpio = df_limpio.sort_index()
        
        return df_limpio
    
    def _calcular_retornos(self, df):
        """
        Calcula retornos diarios y acumulados
        
        Args:
            df (pd.DataFrame): DataFrame con precios
            
        Returns:
            pd.DataFrame: DataFrame con retornos calculados
        """
        self.logger.debug("Calculando retornos...")
        
        # Asegurarse que 'close' existe
        if 'close' not in df.columns:
            self.logger.error("No se encontró la columna 'close' para calcular retornos")
            raise ValueError("No se encontró la columna 'close' para calcular retornos")
        
        # Calcular retornos diarios
        df['retorno'] = df['close'].pct_change() * 100  # en porcentaje
        
        # Calcular retornos logaritmicos (utiles para algunas metricas)
        df['retorno_log'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calcular retornos acumulados
        df['retorno_acum'] = (1 + df['retorno']/100).cumprod() - 1
        
        # Rellenar NaN en la primera fila
        df['retorno'] = df['retorno'].fillna(0)
        df['retorno_log'] = df['retorno_log'].fillna(0)
        df['retorno_acum'] = df['retorno_acum'].fillna(0)
        
        return df
    
    def _calcular_indicadores(self, df):
        """
        Calcula indicadores tecnicos
        
        Args:
            df (pd.DataFrame): DataFrame con precios
            
        Returns:
            pd.DataFrame: DataFrame con indicadores tecnicos
        """
        self.logger.debug("Calculando indicadores técnicos...")
        
        # Importar biblioteca ta para indicadores tecnicos
        import ta
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'], 
            window=self.parametros['ventana_rsi']
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.parametros['ventana_macd_lenta'],
            window_fast=self.parametros['ventana_macd_rapida'],
            window_sign=self.parametros['ventana_macd_senal']
        )
        df['macd'] = macd.macd()
        df['macd_senal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Volatilidad (desviación estándar de los retornos)
        df['volatilidad'] = df['retorno'].rolling(
            window=self.parametros['ventana_volatilidad']
        ).std()
        
        # Medias Moviles
        for ventana in self.parametros['ventana_media_movil']:
            df[f'sma_{ventana}'] = ta.trend.SMAIndicator(
                close=df['close'], window=ventana
            ).sma_indicator()
            
        # Media Movil Exponencial
        df['ema_20'] = ta.trend.EMAIndicator(
            close=df['close'], window=20
        ).ema_indicator()
        
        # Bandas de Bollinger
        bollinger = ta.volatility.BollingerBands(
            close=df['close'], window=20, window_dev=2
        )
        df['bb_alto'] = bollinger.bollinger_hband()
        df['bb_bajo'] = bollinger.bollinger_lband()
        df['bb_medio'] = bollinger.bollinger_mavg()
        df['bb_ancho'] = bollinger.bollinger_wband()
        
        # Volumen normalizado
        if 'volume' in df.columns:
            # Normalizar volumen respecto a su media móvil
            df['volumen_rel'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Rellenar valores NaN que pueden aparecer por las ventanas
        # Usamos ffill y bfill para evitar FutureWarning
        df = df.ffill().bfill()
        
        return df
    
    def _normalizar_caracteristicas(self, df):
        """
        Normaliza las caracteristicas para los modelos
        
        Args:
            df (pd.DataFrame): DataFrame con caracteristicas
            
        Returns:
            pd.DataFrame: DataFrame con caracteristicas normalizadas
        """
        self.logger.debug("Normalizando características...")
        
        # Columnas a normalizar
        cols_to_scale = [
            'retorno', 'rsi', 'macd', 'macd_diff', 'volatilidad',
            'volumen_rel', 'bb_ancho'
        ]
        
        # Verificar qué columnas están disponibles
        cols_to_scale = [col for col in cols_to_scale if col in df.columns]
        
        if not cols_to_scale:
            self.logger.warning("No hay columnas para normalizar")
            return df
        
        # Hacer una copia del DataFrame
        df_norm = df.copy()
        
        # Escalar las características
        scaled_features = self.scaler.fit_transform(df[cols_to_scale])
        
        # Reemplazar los valores originales con los normalizados
        df_norm[cols_to_scale] = scaled_features
        
        return df_norm
    
    def _detectar_anomalias_umbral(self, df):
        """
        Deteccion preliminar de anomalias basada en umbrales
        
        Args:
            df (pd.DataFrame): DataFrame con datos
            
        Returns:
            pd.DataFrame: DataFrame con columna de anomalia agregada
        """
        self.logger.debug("Detectando anomalías por umbral...")
        
        # Marcar como anomalías los días con retornos extremos
        umbral_retorno = self.parametros['anomalia_umbral_retorno']
        df['anomalia_retorno'] = (abs(df['retorno']) > umbral_retorno).astype(int)
        
        # Marcar anomalías de volumen (si la columna existe)
        if 'volumen_rel' in df.columns:
            umbral_volumen = self.parametros['anomalia_umbral_volumen']
            df['anomalia_volumen'] = (df['volumen_rel'] > umbral_volumen).astype(int)
            
            # Anomalia combinada: retorno extremo o volumen extremo
            df['anomalia'] = ((df['anomalia_retorno'] == 0) | (df['anomalia_volumen'] == 0)).astype(int)
        else:
            df['anomalia'] = df['anomalia_retorno']
        
        return df
    
    def discretizar(self, df):
        """
        Discretiza variables para mineria de reglas de asociacion
        
        Args:
            df (pd.DataFrame): DataFrame con datos continuos
            
        Returns:
            pd.DataFrame: DataFrame con variables discretizadas
        """
        self.logger.info("Discretizando variables para reglas de asociación")
        
        df_disc = df.copy()
        
        # Discretizar RSI
        if 'rsi' in df.columns:
            df_disc['rsi_cat'] = pd.cut(
                df['rsi'],
                bins=[0, 30, 70, 100],
                labels=['bajo', 'medio', 'alto']
            )
        
        # Discretizar retornos
        if 'retorno' in df.columns:
            df_disc['retorno_cat'] = pd.cut(
                df['retorno'],
                bins=[-np.inf, -3, -1, 1, 3, np.inf],
                labels=['muy_negativo', 'negativo', 'neutral', 'positivo', 'muy_positivo']
            )
        
        # Discretizar volatilidad
        if 'volatilidad' in df.columns:
            # Calcular percentiles para discretizar
            min_val = df['volatilidad'].min()
            perc_25 = max(min_val + 0.001, df['volatilidad'].quantile(0.25))
            perc_75 = max(perc_25 + 0.001, df['volatilidad'].quantile(0.75))
            
            self.logger.debug(f"Bins para volatilidad: [{min_val}, {perc_25}, {perc_75}, inf]")
            
            # Asegurarnos que los bins aumentan monótonamente
            try:
                df_disc['volatilidad_cat'] = pd.cut(
                    df['volatilidad'],
                    bins=[min_val, perc_25, perc_75, np.inf],
                    labels=['baja', 'media', 'alta']
                )
            except ValueError as e:
                self.logger.warning(f"Error al discretizar volatilidad: {str(e)}. Usando método alternativo.")
                # Alternativa en caso de error: usar qcut (cortes basados en quantiles)
                df_disc['volatilidad_cat'] = pd.qcut(
                    df['volatilidad'],
                    q=[0, 0.25, 0.75, 1.0],
                    labels=['baja', 'media', 'alta'],
                    duplicates='drop'  # Manejar valores duplicados
                )
        
        # Discretizar volumen relativo
        if 'volumen_rel' in df.columns:
            try:
                # Asegurarnos que tenemos valores en el dataframe
                if df['volumen_rel'].isna().all():
                    self.logger.warning("Todos los valores de volumen_rel son NA, saltando discretización")
                    df_disc['volumen_cat'] = pd.Series('normal', index=df.index)
                else:
                    df_disc['volumen_cat'] = pd.cut(
                        df['volumen_rel'],
                        bins=[0, 0.8, 1.2, np.inf],
                        labels=['bajo', 'normal', 'alto']
                    )
            except Exception as e:
                self.logger.warning(f"Error al discretizar volumen: {str(e)}. Usando valores por defecto.")
                df_disc['volumen_cat'] = pd.Series('normal', index=df.index)
        
        # Tendencia basada en media móvil
        if 'close' in df.columns and 'sma_50' in df.columns:
            df_disc['tendencia'] = np.where(
                df['close'] > df['sma_50'], 'alcista',
                np.where(df['close'] < df['sma_50'], 'bajista', 'lateral')
            )
        
        # Señal MACD
        if 'macd' in df.columns and 'macd_senal' in df.columns:
            df_disc['macd_señal'] = np.where(
                df['macd'] > df['macd_senal'], 'compra',
                np.where(df['macd'] < df['macd_senal'], 'venta', 'neutral')
            )
        
        # Discretizar resultado al día siguiente (para reglas de predicción)
        if 'retorno' in df.columns:
            # Desplazar los retornos un día hacia atrás para obtener el retorno del día siguiente
            df_disc['proximo_retorno'] = df['retorno'].shift(-1)
            
            # Discretizar el retorno del día siguiente
            df_disc['proximo_retorno_cat'] = pd.cut(
                df_disc['proximo_retorno'],
                bins=[-np.inf, -1, 1, np.inf],
                labels=['baja', 'neutral', 'sube']
            )
        
        # Rellenar valores nulos en categorías
        categorias = [col for col in df_disc.columns if col.endswith('_cat') or col in ['tendencia', 'macd_señal']]
        for col in categorias:
            # Para categorias, usamos el valor más frecuente
            if df_disc[col].dtype.name == 'category':
                modo = df_disc[col].mode()[0]
                df_disc[col] = df_disc[col].fillna(modo)
        
        # Eliminar la última fila que tendrá NaN en proximo_retorno
        df_disc = df_disc.dropna(subset=['proximo_retorno_cat'])
        
        self.logger.info(f"Discretización completada: {len(categorias)} variables categorizadas")
        
        return df_disc
    
    def guardar_datos(self, df, ruta_archivo):
        """
        Guarda los datos en un archivo CSV
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            ruta_archivo (str): Ruta donde guardar el archivo
        """
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Guardar el DataFrame
        df.to_csv(ruta_archivo)
        self.logger.info(f"Datos guardados en {ruta_archivo}")
    
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
        
        self.logger.info(f"Cargando datos desde {ruta_archivo}")
        df = pd.read_csv(ruta_archivo, index_col=0)
        
        # Convertir el índice a datetime si es una fecha
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
        
        # Convertir columnas categoricas de nuevo a categorias
        for col in df.columns:
            if col.endswith('_cat') or col in ['tendencia', 'macd_señal']:
                df[col] = df[col].astype('category')
                
        return df