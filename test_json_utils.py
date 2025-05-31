"""
Script principal simplificado para probar la funcionalidad de formateo JSON
"""

import sys
import os
import json
import numpy as np
import logging
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Clase NumpyEncoder para serializar objetos NumPy
class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON personalizado que maneja tipos de NumPy
    """
    def default(self, obj):
        # Manejar todos los tipos numéricos de NumPy
        if isinstance(obj, np.number):
            return self._handle_numpy_number(obj)
        # Manejar arrays NumPy
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Manejar booleanos NumPy
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Manejar otros tipos especiales
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)
    
    def _handle_numpy_number(self, obj):
        """Convierte números NumPy a tipos Python nativos"""
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            return complex(obj)
        return obj

def convert_numpy_types(obj):
    """
    Convierte tipos NumPy a tipos Python nativos recursivamente en un objeto.
    """
    # Detectar el tipo de objeto y convertirlo según sea necesario
    if isinstance(obj, np.number):
        # Manejar números NumPy (enteros, flotantes, complejos)
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            return complex(obj)
        return obj
    elif isinstance(obj, np.ndarray):
        # Convertir arrays NumPy a listas Python
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        # Convertir booleanos NumPy
        return bool(obj)
    elif isinstance(obj, (np.datetime64, np.timedelta64)):
        # Convertir tipos de fecha y tiempo NumPy
        return str(obj)
    elif isinstance(obj, dict):
        # Procesar cada elemento del diccionario recursivamente
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Procesar cada elemento de la lista recursivamente
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        # Procesar cada elemento de la tupla recursivamente
        return tuple(convert_numpy_types(item) for item in obj)
    # Mantener otros tipos sin cambios
    return obj

def format_eval_results(results):
    """
    Formatea los resultados de evaluación para una mejor presentación en logs o consola.
    """
    if not results:
        return "No hay resultados disponibles"
    
    # Convertir todos los tipos numpy a tipos Python nativos
    results = convert_numpy_types(results)
    
    # Variable para almacenar el resultado formateado
    formatted = []
    
    # Formatear resultados de clustering
    if "clustering" in results:
        clustering = results["clustering"]
        formatted.append("RESULTADOS DE CLUSTERING:")
        formatted.append(f"  - Calidad de clusters (Silhouette): {clustering.get('calidad_clusters', 0):.4f}")
        formatted.append(f"  - Número de clusters: {clustering.get('num_clusters', 0)}")
        
        # Formatear distribución de clusters
        if "distribucion" in clustering:
            formatted.append("  - Distribución de clusters:")
            for cluster_id, data in clustering["distribucion"].items():
                formatted.append(f"    * {cluster_id}: {data['tamaño']} muestras ({data['porcentaje']:.2f}%), retorno promedio: {data['retorno_promedio']:.4f}")
        formatted.append("")
    
    # Formatear resultados de detección de anomalías
    if "anomalias" in results:
        anomalias = results["anomalias"]
        formatted.append("RESULTADOS DE DETECCIÓN DE ANOMALÍAS:")
        formatted.append(f"  - Precisión: {anomalias.get('precision', 0):.4f}")
        formatted.append(f"  - Recall: {anomalias.get('recall', 0):.4f}")
        formatted.append(f"  - F1-Score: {anomalias.get('f1', 0):.4f}")
        formatted.append(f"  - Anomalías detectadas: {anomalias.get('num_anomalias', 0)} ({anomalias.get('porcentaje', 0):.2f}%)")
        formatted.append("")
    
    # Formatear resultados de reglas de asociación
    if "reglas" in results:
        reglas = results["reglas"]
        formatted.append("RESULTADOS DE REGLAS DE ASOCIACIÓN:")
        formatted.append(f"  - Total de reglas: {reglas.get('total_reglas', 0)}")
        formatted.append(f"  - Confianza promedio: {reglas.get('confianza_promedio', 0):.4f}")
        formatted.append(f"  - Lift promedio: {reglas.get('lift_promedio', 0):.4f}")
        formatted.append(f"  - Reglas predictivas: {reglas.get('reglas_predictivas', 0)}")
        formatted.append("")
    
    # Formatear resultados de rentabilidad
    if "rentabilidad" in results:
        rent = results["rentabilidad"]
        formatted.append("RESULTADOS DE RENTABILIDAD:")
        formatted.append(f"  - Buy & Hold: {rent.get('buy_hold', 0):.2f}%")
        formatted.append(f"  - Estrategia SMA: {rent.get('estrategia_sma', 0):.2f}%")
        formatted.append(f"  - Estrategia Anomalías: {rent.get('estrategia_anomalias', 0):.2f}%")
        formatted.append(f"  - Trades realizados: {rent.get('trades_realizados', 0)}")
    
    return "\n".join(formatted)

def main():
    """
    Función principal para probar la funcionalidad
    """
    # Crear datos de prueba con tipos NumPy
    resultados = {
        'clustering': {
            'calidad_clusters': np.float64(0.75),
            'num_clusters': np.int64(4),
            'distribucion': {
                'Cluster 0': {
                    'tamaño': np.int64(100),
                    'porcentaje': np.float32(40.0),
                    'retorno_promedio': np.float64(0.05)
                }
            }
        },
        'anomalias': {
            'precision': np.float64(0.85),
            'recall': np.float64(0.78),
            'f1': np.float64(0.81),
            'num_anomalias': np.int32(15),
            'porcentaje': np.float32(5.2)
        },
        'reglas': {
            'total_reglas': np.int64(20),
            'confianza_promedio': np.float64(0.72),
            'lift_promedio': np.float64(2.5),
            'reglas_predictivas': np.int64(8)
        },
        'rentabilidad': {
            'buy_hold': np.float64(10.5),
            'estrategia_sma': np.float64(15.2),
            'estrategia_anomalias': np.float64(18.7),
            'trades_realizados': np.int64(42)
        }
    }
    
    logger.info("Probando la funcionalidad de formateo JSON con tipos NumPy")
    
    # 1. Probar codificación JSON
    try:
        json_str = json.dumps(resultados, cls=NumpyEncoder, indent=2)
        logger.info("1. Serialización JSON exitosa")
    except Exception as e:
        logger.error(f"Error en serialización JSON: {str(e)}")
        return
    
    # 2. Probar conversión de tipos NumPy
    try:
        resultados_convertidos = convert_numpy_types(resultados)
        logger.info("2. Conversión de tipos NumPy exitosa")
    except Exception as e:
        logger.error(f"Error en conversión de tipos: {str(e)}")
        return
      # 3. Probar formateo de resultados
    try:
        formatted_results = format_eval_results(resultados)
        logger.info("3. Formateo de resultados exitoso")
        print("\n=== RESULTADOS FORMATEADOS ===")
        print(formatted_results)
        print("==============================\n")
    except Exception as e:
        logger.error(f"Error en formateo de resultados: {str(e)}")
        return
    
    print("\nTodas las pruebas completadas con éxito!\n")

if __name__ == "__main__":
    main()
