#!/usr/bin/env python
"""
Script principal para ejecutar el sistema de minería de datos para trading de criptomonedas
Este script organiza y facilita la ejecución del flujo de trabajo
"""

import os
import sys
import argparse
import logging
import shutil
from datetime import datetime

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar configuración
import config

# Configurar logging
def configurar_logging():
    """Configura el sistema de logs"""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def procesar_argumentos():
    """Procesa los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema de Minería de Datos para Trading de Criptomonedas"
    )
    
    # Argumentos principales
    parser.add_argument(
        '--modo', 
        choices=['completo', 'extraccion', 'preprocesamiento', 'modelado', 
                'evaluacion', 'interfaz', 'test-api', 'limpiar'],
        default='completo', 
        help='Modo de ejecución del sistema'
    )
    
    # Opciones adicionales
    parser.add_argument(
        '--sin-api', 
        action='store_true',
        help='Forzar uso de datos sintéticos sin intentar conexión a API'
    )
    
    parser.add_argument(
        '--forzar-extraccion', 
        action='store_true',
        help='Forzar extracción de datos aunque exista el archivo de datos'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Activar modo debug con información detallada'
    )
    
    parser.add_argument(
        '--limpiar-cache', 
        action='store_true',
        help='Limpiar archivos de caché (__pycache__) antes de ejecutar'
    )
    
    return parser.parse_args()

def limpiar_cache():
    """Limpia archivos de caché del proyecto"""
    logger = logging.getLogger(__name__)
    logger.info("Limpiando archivos de caché...")
    
    directorios_cache = []
    
    # Buscar directorios __pycache__
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for dir in dirs:
            if dir == "__pycache__":
                directorios_cache.append(os.path.join(root, dir))
    
    # Eliminar cada directorio de caché
    for directorio in directorios_cache:
        try:
            shutil.rmtree(directorio)
            logger.info(f"Directorio eliminado: {directorio}")
        except Exception as e:
            logger.error(f"Error al eliminar {directorio}: {str(e)}")
    
    logger.info(f"Se eliminaron {len(directorios_cache)} directorios de caché")

def ejecutar():
    """Función principal que ejecuta el programa según los argumentos"""
    # Procesar argumentos
    args = procesar_argumentos()
    
    # Configurar nivel de logging según modo debug
    if args.debug:
        config.LOG_LEVEL = logging.DEBUG
    
    # Configurar logging
    logger = configurar_logging()
    
    # Mostrar información inicial
    logger.info("==== Sistema de Minería de Datos para Trading de Criptomonedas ====")
    logger.info(f"Versión: 1.0.0")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modo: {args.modo}")
    
    # Limpiar caché si se solicita
    if args.limpiar_cache:
        limpiar_cache()
    
    try:
        # Ejecutar el modo seleccionado
        if args.modo == 'limpiar':
            limpiar_cache()
            logger.info("Limpieza de caché completada")
        elif args.modo == 'test-api':
            import test_basescan_api
            test_basescan_api.main()
            logger.info("Prueba de API completada. Ver api_test.log para detalles")
        else:
            # Importar el archivo principal mejorado
            from main_mejorado import ejecutar_flujo_completo, ejecutar_modo_especifico, iniciar_interfaz
            
            if args.modo == 'completo':
                datos = ejecutar_flujo_completo(args, logger)
                logger.info("Flujo completo ejecutado con éxito")
                logger.info("Iniciando interfaz gráfica...")
                iniciar_interfaz(datos)
            elif args.modo == 'interfaz':
                from src.interfaz.gui import iniciar_interfaz
                logger.info("Iniciando interfaz gráfica...")
                iniciar_interfaz()
            else:
                datos = ejecutar_modo_especifico(args, logger)
                
                # Preguntar si se desea iniciar la interfaz gráfica
                if args.modo in ['extraccion', 'preprocesamiento', 'modelado', 'evaluacion']:
                    try:
                        respuesta = input("¿Desea iniciar la interfaz gráfica con los datos generados? (s/n): ")
                        if respuesta.lower() in ['s', 'si', 'y', 'yes']:
                            logger.info("Iniciando interfaz gráfica con datos generados...")
                            from src.interfaz.gui import iniciar_interfaz
                            iniciar_interfaz(datos)
                    except Exception as e:
                        logger.error(f"Error al iniciar interfaz: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("Ejecución finalizada con éxito")

if __name__ == "__main__":
    ejecutar()
