"""
Modulo para mineria de reglas de asociacion en datos de criptomonedas
"""

import os
import logging
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

class MineroReglas:
    """
    Clase para la mineria de reglas de asociacion en datos de criptomonedas
    """
    
    def __init__(self, soporte_min=0.1, confianza_min=0.7, lift_min=1.2):
        """
        Inicializa el minero de reglas de asociacion
        
        Args:
            soporte_min (float): Soporte minimo para las reglas (default: 0.1)
            confianza_min (float): Confianza minima para las reglas (default: 0.7)
            lift_min (float): Lift minimo para las reglas (default: 1.2)
        """
        self.soporte_min = soporte_min
        self.confianza_min = confianza_min
        self.lift_min = lift_min
        self.logger = logging.getLogger(__name__)
    
    def extraer_reglas(self, df):
        """
        Extrae reglas de asociacion de los datos
        
        Args:
            df (pd.DataFrame): DataFrame con variables discretizadas
            
        Returns:
            pd.DataFrame: DataFrame con reglas de asociacion
        """
        self.logger.info(f"Extrayendo reglas con soporte>={self.soporte_min}, confianza>={self.confianza_min}")
        
        # Verificar que hay columnas categóricas
        col_categoricas = [c for c in df.columns if c.endswith('_cat') 
                          or c in ['tendencia', 'macd_señal']]
        
        if not col_categoricas:
            self.logger.error("No se encontraron columnas categóricas para extraer reglas")
            raise ValueError("No se encontraron columnas categóricas para extraer reglas")
        
        # Crear matriz booleana para apriori
        df_preparado = self._preparar_datos(df, col_categoricas)
        
        # Encontrar conjuntos frecuentes
        self.logger.debug(f"Buscando conjuntos frecuentes con soporte>={self.soporte_min}")
        conjuntos_frecuentes = apriori(
            df_preparado, 
            min_support=self.soporte_min,
            use_colnames=True
        )
        
        if len(conjuntos_frecuentes) == 0:
            self.logger.warning(f"No se encontraron conjuntos frecuentes con soporte>={self.soporte_min}")
            return pd.DataFrame()
        
        self.logger.debug(f"Se encontraron {len(conjuntos_frecuentes)} conjuntos frecuentes")
        
        # Extraer reglas de asociacion
        self.logger.debug(f"Extrayendo reglas con confianza>={self.confianza_min}")
        reglas = association_rules(
            conjuntos_frecuentes, 
            metric="confidence", 
            min_threshold=self.confianza_min
        )
        
        # Filtrar por lift
        reglas = reglas[reglas['lift'] >= self.lift_min]
        
        if len(reglas) == 0:
            self.logger.warning(f"No se encontraron reglas con lift>={self.lift_min}")
            return pd.DataFrame()
        
        # Ordenar reglas por lift descendente
        reglas = reglas.sort_values('lift', ascending=False)
        
        # Convertir antecedentes y consecuentes a formato legible
        reglas['antecedentes'] = reglas['antecedents'].apply(lambda x: ', '.join(x))
        reglas['consecuentes'] = reglas['consequents'].apply(lambda x: ', '.join(x))
        
        # Seleccionar columnas relevantes
        reglas_final = reglas[[
            'antecedentes', 'consecuentes', 'support', 'confidence', 'lift', 'leverage', 'conviction'
        ]]
        
        # Renombrar columnas para mejor interpretabilidad
        reglas_final.columns = [
            'antecedentes', 'consecuentes', 'soporte', 'confianza', 'lift', 'leverage', 'conviccion'
        ]
        
        self.logger.info(f"Extraidas {len(reglas_final)} reglas de asociacion")
        
        return reglas_final
    
    def _preparar_datos(self, df, col_categoricas):
        """
        Prepara los datos para la mineria de reglas de asociacion
        
        Args:
            df (pd.DataFrame): DataFrame con variables discretizadas
            col_categoricas (list): Lista de columnas categoricas
            
        Returns:
            pd.DataFrame: DataFrame con matriz booleana para apriori
        """
        # Crear columnas indicadoras para cada valor categorico
        df_preparado = pd.DataFrame()
        
        for col in col_categoricas:
            # Verificar que la columna existe
            if col not in df.columns:
                self.logger.warning(f"Columna {col} no encontrada, se omite")
                continue
            
            # Obtener todos los valores únicos de la columna
            valores = df[col].dropna().unique()
            
            for val in valores:
                # Crear columna booleana para cada valor
                nombre_col = f"{col}_{val}"
                df_preparado[nombre_col] = (df[col] == val).astype(int)
        
        # Verificar que hay columnas
        if df_preparado.empty:
            self.logger.error("No se pudieron crear columnas indicadoras")
            raise ValueError("No se pudieron crear columnas indicadoras")
        
        self.logger.debug(f"DataFrame preparado con {df_preparado.shape[1]} columnas indicadoras")
        
        return df_preparado
    
    def filtrar_reglas_predictivas(self, reglas):
        """
        Filtra las reglas para quedarse solo con las que predicen movimientos futuros
        
        Args:
            reglas (pd.DataFrame): DataFrame con todas las reglas
            
        Returns:
            pd.DataFrame: DataFrame con reglas filtradas
        """
        # Filtrar reglas donde el consecuente es proximo_retorno
        reglas_pred = reglas[reglas['consecuentes'].str.contains('proximo_retorno')]
        
        if len(reglas_pred) == 0:
            self.logger.warning("No se encontraron reglas predictivas")
            return pd.DataFrame()
        
        self.logger.info(f"Se encontraron {len(reglas_pred)} reglas predictivas")
        
        return reglas_pred
    
    def guardar_reglas(self, reglas, ruta_archivo):
        """
        Guarda las reglas de asociacion en un archivo CSV
        
        Args:
            reglas (pd.DataFrame): DataFrame con reglas de asociacion
            ruta_archivo (str): Ruta donde guardar el archivo
        """
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        
        # Guardar reglas
        reglas.to_csv(ruta_archivo, index=False)
        self.logger.info(f"Reglas guardadas en {ruta_archivo}")
    
    def cargar_reglas(self, ruta_archivo):
        """
        Carga reglas de asociacion desde un archivo CSV
        
        Args:
            ruta_archivo (str): Ruta del archivo a cargar
            
        Returns:
            pd.DataFrame: DataFrame con reglas de asociacion
        """
        if not os.path.exists(ruta_archivo):
            self.logger.error(f"El archivo {ruta_archivo} no existe")
            raise FileNotFoundError(f"El archivo {ruta_archivo} no existe")
        
        self.logger.info(f"Cargando reglas desde {ruta_archivo}")
        reglas = pd.read_csv(ruta_archivo)
        
        return reglas
    
    def evaluar_regla_en_datos(self, regla, df_discretizado):
        """
        Evalua una regla especifica en el conjunto de datos
        
        Args:
            regla (pd.Series): Fila de DataFrame con una regla
            df_discretizado (pd.DataFrame): DataFrame con datos discretizados
            
        Returns:
            dict: Estadisticas de evaluacion de la regla
        """
        # Obtener antecedentes y consecuentes
        antecedentes = regla['antecedentes'].split(', ')
        consecuentes = regla['consecuentes'].split(', ')
        
        # Verificar condiciones de antecedentes en cada fila
        condiciones_ant = []
        for ant in antecedentes:
            col, val = ant.split('_', 1)
            condiciones_ant.append(df_discretizado[col] == val)
        
        # Combinar condiciones con AND
        mask_antecedente = condiciones_ant[0]
        for cond in condiciones_ant[1:]:
            mask_antecedente = mask_antecedente & cond
        
        # Filtrar filas que cumplen antecedente
        filas_cumplen = df_discretizado[mask_antecedente]
        
        # Verificar consecuentes
        condiciones_cons = []
        for cons in consecuentes:
            col, val = cons.split('_', 1)
            condiciones_cons.append(filas_cumplen[col] == val)
        
        # Combinar condiciones con AND
        mask_consecuente = condiciones_cons[0]
        for cond in condiciones_cons[1:]:
            mask_consecuente = mask_consecuente & cond
        
        # Contar aciertos
        total_cumplen_antecedente = mask_antecedente.sum()
        total_aciertos = mask_consecuente.sum() if total_cumplen_antecedente > 0 else 0
        
        # Calcular precision
        precision = total_aciertos / total_cumplen_antecedente if total_cumplen_antecedente > 0 else 0
        
        return {
            'total_casos': total_cumplen_antecedente,
            'aciertos': total_aciertos,
            'precision': precision,
            'confianza_original': regla['confianza'],
            'lift': regla['lift']
        }
