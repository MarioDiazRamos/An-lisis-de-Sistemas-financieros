"""
Archivo principal del sistema de mineria de datos para trading de criptomonedas
Este script orquesta la ejecucion de todos los modulos del proyecto
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Añadir el directorio raíz al path para importar módulos
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

# Importar módulos del proyecto
from src.extraccion_datos.extractor import Extractor
from src.preprocesamiento.preprocesador import Preprocesador
from src.modelos.clustering import ModeloClustering
from src.modelos.anomalias import ModeloAnomalias
from src.modelos.reglas_asociacion import MineroReglas
from src.evaluacion.evaluador import Evaluador
from src.interfaz.gui import iniciar_interfaz

def procesar_argumentos():
    """Procesa los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Sistema de Mineria de Datos para Trading de Criptomonedas")
    parser.add_argument('--modo', choices=['completo', 'extraccion', 'preprocesamiento', 'modelado', 'evaluacion', 'interfaz'], 
                      default='completo', help='Modo de ejecución')
    parser.add_argument('--forzar-extraccion', action='store_true', 
                      help='Forzar la extracción de datos aunque exista el archivo de datos')
    parser.add_argument('--debug', action='store_true', 
                      help='Ejecutar en modo debug (más información en los logs)')
    
    return parser.parse_args()

def ejecutar_flujo_completo(args, logger):
    """Ejecuta el flujo completo del proceso de minería de datos"""
    logger.info("Iniciando ejecución del flujo completo")
    
    # 1. Extracción de datos
    extractor = Extractor(config.API_KEY, config.API_URL, config.API_RATE_LIMIT)
    
    if args.forzar_extraccion or not os.path.exists(config.DATOS_CRUDOS):
        logger.info("Extrayendo datos de la API...")
        df_raw = extractor.extraer_datos_historicos(config.FECHA_INICIO, config.FECHA_FIN)
        extractor.guardar_datos(df_raw, config.DATOS_CRUDOS)
        logger.info(f"Datos guardados en {config.DATOS_CRUDOS}")
    else:
        logger.info(f"Usando datos existentes de {config.DATOS_CRUDOS}")
        df_raw = extractor.cargar_datos(config.DATOS_CRUDOS)
    
    # 2. Preprocesamiento
    logger.info("Preprocesando datos...")
    preprocesador = Preprocesador(config.PARAMETROS)
    df_procesado = preprocesador.procesar(df_raw)
    df_discretizado = preprocesador.discretizar(df_procesado)
    
    preprocesador.guardar_datos(df_procesado, config.DATOS_PROCESADOS)
    preprocesador.guardar_datos(df_discretizado, config.DATOS_DISCRETIZADOS)
    logger.info(f"Datos procesados guardados en {config.DATOS_PROCESADOS}")
    
    # 3. Modelado - Clustering
    logger.info("Aplicando clustering...")
    modelo_clustering = ModeloClustering(
        n_clusters=config.PARAMETROS['clustering_num_clusters'],
        random_state=config.PARAMETROS['clustering_random_state']
    )
    df_clusters = modelo_clustering.entrenar_y_predecir(df_procesado)
    modelo_clustering.guardar_modelo(config.MODELO_CLUSTERING)
    
    # 4. Modelado - Detección de anomalías
    logger.info("Entrenando detector de anomalías...")
    modelo_anomalias = ModeloAnomalias(
        n_estimators=config.PARAMETROS['rf_num_arboles'],
        max_depth=config.PARAMETROS['rf_max_depth'],
        random_state=config.PARAMETROS['rf_random_state']
    )
    df_anomalias = modelo_anomalias.entrenar_y_predecir(df_procesado)
    modelo_anomalias.guardar_modelo(config.MODELO_ANOMALIAS)
    
    # 5. Minería de reglas de asociación
    logger.info("Extrayendo reglas de asociación...")
    minero_reglas = MineroReglas(
        soporte_min=config.PARAMETROS['reglas_soporte_min'],
        confianza_min=config.PARAMETROS['reglas_confianza_min'],
        lift_min=config.PARAMETROS['reglas_lift_min']
    )
    reglas = minero_reglas.extraer_reglas(df_discretizado)
    minero_reglas.guardar_reglas(reglas, config.REGLAS_ASOCIACION)
    
    # 6. Evaluación
    logger.info("Evaluando modelos...")
    evaluador = Evaluador(config.PARAMETROS)
    resultados = evaluador.evaluar_modelos(df_procesado, df_clusters, df_anomalias, reglas)
    
    # 7. Mostrar resultados por consola
    for modelo, metricas in resultados.items():
        logger.info(f"Resultados de {modelo}:")
        for metrica, valor in metricas.items():
            logger.info(f"  {metrica}: {valor}")
    
    return {
        'df_raw': df_raw,
        'df_procesado': df_procesado,
        'df_discretizado': df_discretizado,
        'df_clusters': df_clusters,
        'df_anomalias': df_anomalias,
        'reglas': reglas,
        'resultados': resultados
    }

def ejecutar_modo_especifico(args, logger):
    """Ejecuta solo una parte específica del proceso"""
    if args.modo == 'extraccion':
        extractor = Extractor(config.API_KEY, config.API_URL, config.API_RATE_LIMIT)
        df_raw = extractor.extraer_datos_historicos(config.FECHA_INICIO, config.FECHA_FIN)
        extractor.guardar_datos(df_raw, config.DATOS_CRUDOS)
        logger.info(f"Extracción completada. Datos guardados en {config.DATOS_CRUDOS}")
        
    elif args.modo == 'preprocesamiento':
        extractor = Extractor(config.API_KEY, config.API_URL, config.API_RATE_LIMIT)
        df_raw = extractor.cargar_datos(config.DATOS_CRUDOS)
        
        preprocesador = Preprocesador(config.PARAMETROS)
        df_procesado = preprocesador.procesar(df_raw)
        df_discretizado = preprocesador.discretizar(df_procesado)
        
        preprocesador.guardar_datos(df_procesado, config.DATOS_PROCESADOS)
        preprocesador.guardar_datos(df_discretizado, config.DATOS_DISCRETIZADOS)
        logger.info(f"Preprocesamiento completado. Datos guardados.")
        
    elif args.modo == 'modelado':
        preprocesador = Preprocesador(config.PARAMETROS)
        df_procesado = preprocesador.cargar_datos(config.DATOS_PROCESADOS)
        df_discretizado = preprocesador.cargar_datos(config.DATOS_DISCRETIZADOS)
        
        # Clustering
        modelo_clustering = ModeloClustering(
            n_clusters=config.PARAMETROS['clustering_num_clusters'],
            random_state=config.PARAMETROS['clustering_random_state']
        )
        df_clusters = modelo_clustering.entrenar_y_predecir(df_procesado)
        modelo_clustering.guardar_modelo(config.MODELO_CLUSTERING)
        
        # Anomalías
        modelo_anomalias = ModeloAnomalias(
            n_estimators=config.PARAMETROS['rf_num_arboles'],
            max_depth=config.PARAMETROS['rf_max_depth'],
            random_state=config.PARAMETROS['rf_random_state']
        )
        df_anomalias = modelo_anomalias.entrenar_y_predecir(df_procesado)
        modelo_anomalias.guardar_modelo(config.MODELO_ANOMALIAS)
        
        # Reglas
        minero_reglas = MineroReglas(
            soporte_min=config.PARAMETROS['reglas_soporte_min'],
            confianza_min=config.PARAMETROS['reglas_confianza_min'],
            lift_min=config.PARAMETROS['reglas_lift_min']
        )
        reglas = minero_reglas.extraer_reglas(df_discretizado)
        minero_reglas.guardar_reglas(reglas, config.REGLAS_ASOCIACION)
        
        logger.info(f"Modelado completado. Modelos guardados.")
        
    elif args.modo == 'evaluacion':
        preprocesador = Preprocesador(config.PARAMETROS)
        df_procesado = preprocesador.cargar_datos(config.DATOS_PROCESADOS)
        
        modelo_clustering = ModeloClustering()
        modelo_clustering.cargar_modelo(config.MODELO_CLUSTERING)
        df_clusters = modelo_clustering.predecir(df_procesado)
        
        modelo_anomalias = ModeloAnomalias()
        modelo_anomalias.cargar_modelo(config.MODELO_ANOMALIAS)
        df_anomalias = modelo_anomalias.predecir(df_procesado)
        
        minero_reglas = MineroReglas()
        reglas = minero_reglas.cargar_reglas(config.REGLAS_ASOCIACION)
        
        evaluador = Evaluador(config.PARAMETROS)
        resultados = evaluador.evaluar_modelos(df_procesado, df_clusters, df_anomalias, reglas)
        
        for modelo, metricas in resultados.items():
            logger.info(f"Resultados de {modelo}:")
            for metrica, valor in metricas.items():
                logger.info(f"  {metrica}: {valor}")
                
    elif args.modo == 'interfaz':
        logger.info("Iniciando interfaz grafica...")
        iniciar_interfaz()
        
    logger.info(f"Modo {args.modo} completado exitosamente.")

def main():
    """Función principal"""
    # Procesar argumentos
    args = procesar_argumentos()
    
    # Configurar logging
    if args.debug:
        config.LOG_LEVEL = logging.DEBUG
    
    logger = configurar_logging()
    logger.info(f"Iniciando sistema en modo: {args.modo}")
    
    try:
        if args.modo == 'completo':
            datos = ejecutar_flujo_completo(args, logger)
            logger.info("Flujo completo ejecutado con éxito.")
            logger.info("Iniciando interfaz gráfica...")
            iniciar_interfaz(datos)
        elif args.modo == 'interfaz':
            logger.info("Iniciando interfaz gráfica...")
            iniciar_interfaz()
        else:
            ejecutar_modo_especifico(args, logger)
            
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        sys.exit(1)
        
    logger.info("Ejecución finalizada con éxito.")

if __name__ == "__main__":
    main()
