"""
Utilidades para manejo de JSON con tipos NumPy
"""

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON personalizado que maneja tipos de NumPy
    """
    def default(self, obj):
        # Manejar todos los tipos de enteros NumPy
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        # Manejar todos los tipos de punto flotante NumPy
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        # Manejar arrays NumPy
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Manejar booleanos NumPy
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        # Manejar complejos NumPy
        elif isinstance(obj, np.complex64) or isinstance(obj, np.complex128):
            return complex(obj)
        # Manejar otros tipos especiales
        elif isinstance(obj, (np.void, np.datetime64, np.timedelta64)):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """
    Convierte tipos NumPy a tipos Python nativos recursivamente en un objeto.
    
    Args:
        obj: Objeto que puede contener tipos NumPy (int, float, bool, ndarray)
        
    Returns:
        Objeto con tipos NumPy convertidos a tipos Python nativos
    """
    # Manejar casos de NumPy scalar
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.complex64) or isinstance(obj, np.complex128):
        return complex(obj)
    elif isinstance(obj, (np.void, np.datetime64, np.timedelta64)):
        return str(obj)
    # Manejar estructuras de datos
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    # Si no es un tipo especial, devolver como está
    else:
        return obj

def format_eval_results(results):
    """
    Formatea los resultados de evaluación para una mejor presentación en logs o consola.
    
    Args:
        results (dict): Diccionario con resultados de evaluación
        
    Returns:
        str: Resultados formateados como texto
    """
    if not results:
        return "No hay resultados disponibles"
    
    # Convertir todos los tipos numpy a tipos Python nativos
    results = convert_numpy_types(results)
    
    # Variable para almacenar el resultado formateado
    formatted = []
    
    # Formatear resultados de clustering
    if 'clustering' in results:
        clustering = results['clustering']
        formatted.append("RESULTADOS DE CLUSTERING:")
        formatted.append(f"  - Calidad de clusters (Silhouette): {clustering.get('calidad_clusters', 0):.4f}")
        formatted.append(f"  - Número de clusters: {clustering.get('num_clusters', 0)}")
        
        # Formatear distribución de clusters
        if 'distribucion' in clustering:
            formatted.append("  - Distribución de clusters:")
            for cluster_id, data in clustering['distribucion'].items():
                formatted.append(f"    * {cluster_id}: {data['tamaño']} muestras ({data['porcentaje']:.2f}%), "
                               f"retorno promedio: {data['retorno_promedio']:.4f}")
        formatted.append("")
    
    # Formatear resultados de detección de anomalías
    if 'anomalias' in results:
        anomalias = results['anomalias']
        formatted.append("RESULTADOS DE DETECCIÓN DE ANOMALÍAS:")
        formatted.append(f"  - Precisión: {anomalias.get('precision', 0):.4f}")
        formatted.append(f"  - Recall: {anomalias.get('recall', 0):.4f}")
        formatted.append(f"  - F1-Score: {anomalias.get('f1', 0):.4f}")
        formatted.append(f"  - Anomalías detectadas: {anomalias.get('num_anomalias', 0)} ({anomalias.get('porcentaje', 0):.2f}%)")
        formatted.append("")
    
    # Formatear resultados de reglas de asociación
    if 'reglas' in results:
        reglas = results['reglas']
        formatted.append("RESULTADOS DE REGLAS DE ASOCIACIÓN:")
        formatted.append(f"  - Total de reglas: {reglas.get('total_reglas', 0)}")
        formatted.append(f"  - Confianza promedio: {reglas.get('confianza_promedio', 0):.4f}")
        formatted.append(f"  - Lift promedio: {reglas.get('lift_promedio', 0):.4f}")
        formatted.append(f"  - Reglas predictivas: {reglas.get('reglas_predictivas', 0)}")
        formatted.append("")
    
    # Formatear resultados de rentabilidad
    if 'rentabilidad' in results:
        rent = results['rentabilidad']
        formatted.append("RESULTADOS DE RENTABILIDAD:")
        formatted.append(f"  - Buy & Hold: {rent.get('buy_hold', 0):.2f}%")
        formatted.append(f"  - Estrategia SMA: {rent.get('estrategia_sma', 0):.2f}%")
        formatted.append(f"  - Estrategia Anomalías: {rent.get('estrategia_anomalias', 0):.2f}%")
        formatted.append(f"  - Trades realizados: {rent.get('trades_realizados', 0)}")
    
    return "\n".join(formatted)
