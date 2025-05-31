"""
Script temporal para probar la corrección del módulo de anomalías
"""
import sys
import os
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Importamos la clase corregida
    logger.info("Intentando importar la clase corregida...")
    from src.modelos.anomalias_corregido import ModeloAnomalias
    
    # Creamos una instancia básica para probar
    logger.info("Creando instancia del modelo...")
    modelo = ModeloAnomalias()
    
    logger.info("La importación y creación del modelo funciona correctamente!")
    print("¡Prueba exitosa! El módulo corregido funciona correctamente.")
    
except Exception as e:
    logger.error(f"Error al probar el módulo corregido: {str(e)}", exc_info=True)
    print(f"Error en la prueba: {str(e)}")
    sys.exit(1)
