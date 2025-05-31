"""
Script temporal para diagnosticar problemas de importación
"""

import sys
import os

# Añadir el directorio raiz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Intentar importaciones
try:
    print("Intentando importar ModeloClustering...")
    from src.modelos.clustering import ModeloClustering
    print("¡Éxito!")
except Exception as e:
    print(f"Error al importar ModeloClustering: {str(e)}")

try:
    print("Intentando importar ModeloAnomalias...")
    with open("src/modelos/anomalias.py", "r") as f:
        content = f.read()
        print(f"Longitud del archivo anomalias.py: {len(content)} caracteres")
        print("Primeras 100 caracteres:")
        print(content[:100])
    
    # Intentar diagnosticar el problema
    import ast
    try:
        ast.parse(content)
        print("El archivo es Python válido!")
    except SyntaxError as e:
        print(f"Error de sintaxis en línea {e.lineno}, columna {e.offset}: {e.text}")
        
    # Intentar ejecutar el código manualmente
    namespace = {}
    try:
        exec(content, namespace)
        print("Ejecutado correctamente")
        if "ModeloAnomalias" in namespace:
            print("La clase ModeloAnomalias está definida correctamente!")
        else:
            print("Error: ModeloAnomalias no está definida en el espacio de nombres")
    except Exception as e:
        print(f"Error al ejecutar: {str(e)}")
        
except Exception as e:
    print(f"Error al leer archivos: {str(e)}")
