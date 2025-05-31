import json
import numpy as np
from tabulate import tabulate

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON personalizado que maneja tipos de NumPy
    """
    def default(self, obj):
        if isinstance(obj, np.number):
            return self._handle_numpy_number(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        return super().default(obj)

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
    if isinstance(obj, np.number):
        if np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.complexfloating):
            return complex(obj)
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def format_eval_results(results):
    """
    Formatea los resultados de evaluación como tablas para una mejor presentación.
    
    Args:
        results (dict): Diccionario con resultados de evaluación
        
    Returns:
        str: Resultados formateados como tablas
    """
    if not results:
        return "No hay resultados disponibles"
    
    # Convertir todos los tipos NumPy a tipos Python nativos
    results = convert_numpy_types(results)
    
    # Lista para almacenar las secciones formateadas
    formatted = []
    
    # Formatear resultados de clustering
    if "clustering" in results:
        clustering = results["clustering"]
        formatted.append("RESULTADOS DE CLUSTERING")
        general_data = [
            ["Silhouette Score", f"{clustering.get('calidad_clusters', 0):.4f}"],
            ["Número de Clusters", clustering.get('num_clusters', 0)]
        ]
        formatted.append(tabulate(general_data, headers=["Métrica", "Valor"], tablefmt="grid"))
        
        if "distribucion" in clustering:
            cluster_data = []
            for cluster_id, data in clustering["distribucion"].items():
                cluster_data.append([
                    cluster_id,
                    data['tamaño'],
                    f"{data['porcentaje']:.2f}%",
                    f"{data['retorno_promedio']:.4f}",
                    f"{data.get('porc_anomalias', 0):.2f}%"
                ])
            headers = ["Cluster", "Tamaño", "Porcentaje", "Retorno Promedio", "Anomalías (%)"]
            formatted.append("\nDistribución de Clusters")
            formatted.append(tabulate(cluster_data, headers=headers, tablefmt="grid"))
        formatted.append("")
    
    # Formatear resultados de detección de anomalías
    if "anomalias" in results:
        anomalias = results["anomalias"]
        formatted.append("RESULTADOS DE DETECCIÓN DE ANOMALÍAS")
        anomaly_data = [
            ["Precisión", f"{anomalias.get('precision', 0):.4f}"],
            ["Recall", f"{anomalias.get('recall', 0):.4f}"],
            ["F1-Score", f"{anomalias.get('f1', 0):.4f}"],
            ["Anomalías Detectadas", f"{anomalias.get('num_anomalias', 0)} ({anomalias.get('porcentaje', 0):.2f}%)"]
        ]
        formatted.append(tabulate(anomaly_data, headers=["Métrica", "Valor"], tablefmt="grid"))
        formatted.append("")
    
    # Formatear resultados de reglas de asociación
    if "reglas" in results:
        reglas = results["reglas"]
        formatted.append("RESULTADOS DE REGLAS DE ASOCIACIÓN")
        rules_data = [
            ["Total de Reglas", reglas.get('total_reglas', 0)],
            ["Confianza Promedio", f"{reglas.get('confianza_promedio', 0):.4f}"],
            ["Lift Promedio", f"{reglas.get('lift_promedio', 0):.4f}"],
            ["Reglas Predictivas", reglas.get('reglas_predictivas', 0)]
        ]
        formatted.append(tabulate(rules_data, headers=["Métrica", "Valor"], tablefmt="grid"))
        formatted.append("")
    
    # Formatear resultados de rentabilidad
    if "rentabilidad" in results:
        rent = results["rentabilidad"]
        formatted.append("RESULTADOS DE RENTABILIDAD")
        profitability_data = [
            ["Buy & Hold", f"{rent.get('buy_hold', 0):.2f}%"],
            ["Estrategia SMA", f"{rent.get('estrategia_sma', 0):.2f}%"],
            ["Estrategia Anomalías", f"{rent.get('estrategia_anomalias', 0):.2f}%"],
            ["Trades Anomalías", rent.get('trades_anomalias', 0)],
            ["Trades SMA", rent.get('trades_sma', 0)]
        ]
        formatted.append(tabulate(profitability_data, headers=["Estrategia", "Valor"], tablefmt="grid"))
    
    return "\n".join(formatted)