# Sistema de Minería de Datos para Trading Práctico en el Criptomercado (Bitcoin)

Este proyecto implementa un sistema completo de minería de datos para el criptomercado, enfocado en Bitcoin (BTC), que genera alertas de trading y optimiza decisiones de inversión mediante detección de anomalías y reglas de asociación.

## Descripción

El sistema de minería de datos analiza precios históricos de Bitcoin para:

1. Detectar anomalías y movimientos extremos
2. Agrupar días de trading con características similares (clustering)
3. Descubrir reglas de asociación entre indicadores técnicos y movimientos futuros
4. Evaluar la rentabilidad de estrategias basadas en estas señales

La aplicación es modular, implementada en Python, con una interfaz gráfica que permite visualizar los resultados.

Se utiliza la API de Basescan.org para obtener datos históricos y en tiempo real de los precios de Bitcoin. El sistema puede operar con datos reales o sintéticos (cuando no se dispone de acceso a la API), lo que facilita el desarrollo y las pruebas.

## Estructura del Proyecto Mejorado

```
.
├── config.py                    # Configuración global
├── run.py                       # Punto de entrada mejorado
├── main_mejorado.py             # Script principal de orquestación
├── test_basescan_api.py         # Script para diagnosticar API
├── requirements.txt             # Dependencias
├── datos/                       # Datos almacenados y modelos
├── logs/                        # Archivos de registro
├── src/                         # Código fuente
│   ├── extraccion_datos/        # Módulo para extracción de datos
│   │   └── extractor_mejorado.py   # Versión mejorada con manejo de errores y fallbacks
│   ├── preprocesamiento/        # Módulo para limpieza y preparación de datos
│   ├── modelos/                 # Módulos de algoritmos de minería de datos
│   │   ├── anomalias_mejorado.py   # Versión mejorada con corrección de warnings
│   │   ├── reglas_asociacion_mejorado.py  # Versión mejorada con tipos booleanos
│   │   └── clustering.py           # Algoritmo de clustering
│   ├── evaluacion/              # Módulo para evaluación de modelos
│   ├── interfaz/                # Interfaz gráfica de usuario
│   └── utils/                   # Utilidades generales
└── tests/                       # Pruebas unitarias
```

## Características

- **Detección de Anomalías**: Identifica días con movimientos inusuales (retornos extremos o volumen anormal)
- **Clustering**: Agrupa días de trading con características similares
- **Reglas de Asociación**: Descubre patrones entre indicadores técnicos y movimientos futuros
- **Evaluación de Estrategias**: Compara la rentabilidad de estrategias basadas en los patrones detectados
- **Interfaz Gráfica**: Visualización de precios, clusters, anomalías y reglas
- **Modularidad**: Diseño modular para fácil mantenimiento y extensión
- **Resistente a Fallos**: Manejo robusto de errores y fallbacks a datos sintéticos
- **Diagnóstico API**: Herramientas para diagnosticar problemas de conectividad

## Mejoras Implementadas

- ✅ **Mejor manejo de errores API**: Fallback a fuentes alternativas cuando Basescan.org no está disponible
- ✅ **Corrección de warnings**: Solución para advertencias de feature_names en RandomForestClassifier
- ✅ **Tipos de datos correctos**: Conversión explícita a booleanos en reglas de asociación
- ✅ **Modo sin API**: Opción `--sin-api` para usar datos sintéticos sin intentar conexiones API
- ✅ **Diagnóstico API**: Script dedicado para probar y diagnosticar problemas de conectividad
- ✅ **Organización del código**: Eliminación de redundancias manteniendo la funcionalidad original

## Requisitos

- Python 3.9+
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar o descargar el repositorio

2. Crear un entorno virtual:
```
python -m venv venv
```

3. Activar el entorno virtual:
   - Windows:
   ```
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```
   source venv/bin/activate
   ```

4. Instalar dependencias:
```
pip install -r requirements.txt
```

5. Configurar parámetros en `config.py` (especialmente la API_KEY para datos externos)

## Uso

Ejecutar la aplicación con el script mejorado:

```
python run.py
```

Opciones disponibles:
- `--modo completo`: Ejecuta todo el flujo (predeterminado)
- `--modo extraccion`: Solo extrae datos
- `--modo preprocesamiento`: Solo preprocesa los datos
- `--modo modelado`: Solo entrena los modelos
- `--modo evaluacion`: Solo evalúa los modelos
- `--modo interfaz`: Solo muestra la interfaz gráfica
- `--modo test-api`: Ejecuta diagnóstico de la API de Basescan
- `--modo limpiar`: Limpia archivos de caché
- `--sin-api`: Usa datos sintéticos sin intentar conexión a API
- `--forzar-extraccion`: Fuerza la extracción de datos aunque exista el archivo
- `--debug`: Muestra información detallada de depuración
- `--limpiar-cache`: Limpia archivos de caché antes de ejecutar

Ejemplos:

```bash
# Ejecutar el flujo completo usando datos sintéticos (sin API)
python run.py --modo completo --sin-api

# Probar la conectividad con la API de Basescan
python run.py --modo test-api

# Solo mostrar la interfaz gráfica
python run.py --modo interfaz

# Extraer datos y forzar nueva extracción
python run.py --modo extraccion --forzar-extraccion

# Ejecutar en modo debug para ver más información
python run.py --debug
```

## Metodología

El proyecto sigue la metodología CRISP-DM para minería de datos:

1. **Comprensión del Negocio**: Trading de criptomonedas
2. **Comprensión de los Datos**: Datos históricos de precios y volúmenes de BTC
3. **Preparación de los Datos**: Limpieza e ingeniería de características
4. **Modelado**: Clustering, detección de anomalías, reglas de asociación
5. **Evaluación**: Métricas de calidad y backtesting de estrategias
6. **Despliegue**: Interfaz gráfica y sistema de alertas

## Métricas de Evaluación

- **Clustering**: Silhouette Score (>0.5)
- **Anomalías**: F1-score (>0.85), precisión, recall
- **Reglas de asociación**: Confianza (>0.7), lift (>1.2)
- **Estrategias de trading**: Rentabilidad vs. buy-and-hold, máximo drawdown

## Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes, abra primero un issue para discutir lo que le gustaría cambiar.

## Licencia

Este proyecto está bajo la Licencia MIT - vea el archivo LICENSE para más detalles.

## Contacto

Autor - [Nombre del Autor](mailto:correo@example.com)

## Integración con Basescan.org API

El sistema está diseñado para utilizar la API de Basescan.org para obtener datos históricos y en tiempo real de Bitcoin. Para configurar la API:

1. Regístrese en [Basescan.org](https://basescan.org) y obtenga una clave API
2. Configure su clave API en `config.py`:
   ```python
   API_KEY = "SU_CLAVE_API_AQUI"
   ```

### Diagnóstico de API

Para diagnosticar problemas con la API de Basescan, utilice:

```bash
python run.py --modo test-api
```

Este comando ejecutará pruebas para verificar:
- Conexión básica a la API
- Validez de la clave API
- Funcionamiento de endpoints específicos
- Disponibilidad de documentación

Los resultados se registran en el archivo `api_test.log`.

### Operación sin API

Si no tiene acceso a la API o experimenta problemas de conectividad, puede utilizar el modo sin API:

```bash
python run.py --sin-api
```

Este modo generará datos sintéticos realistas para probar todo el flujo del sistema.

## Estructura de los Módulos Mejorados

### Extractor de Datos Mejorado
- Implementa múltiples reintentos y espera exponencial
- Manejo detallado de errores HTTP
- Fallback a API alternativa (CoinGecko)
- Generación de datos sintéticos cuando todas las API fallan

### Detector de Anomalías Mejorado
- Corrige warnings sobre feature_names_in_
- Mejor manejo de tipos de datos
- Respaldo para operación en caso de errores

### Minería de Reglas Mejorado
- Conversión explícita a tipos booleanos para evitar advertencias
- Manejo más robusto de errores
- Mejor documentación de funciones

## Conclusiones

Este sistema de minería de datos proporciona una solución modular y robusta para el análisis de datos de criptomonedas, con especial énfasis en la detección de anomalías y descubrimiento de patrones mediante reglas de asociación.

Las mejoras implementadas garantizan:
1. Mayor resistencia a fallos de conectividad
2. Eliminación de advertencias y errores en la ejecución
3. Mayor flexibilidad para operar con o sin acceso a API externa
4. Mejor organización del código y eliminación de redundancias

El sistema está diseñado para ser extensible, permitiendo la adición de nuevos algoritmos de minería de datos, indicadores técnicos y fuentes de datos en el futuro.
   API_KEY = "SU_CLAVE_API_AQUI"
   ```

El sistema utiliza los siguientes endpoints de la API:
- `/v1/klines` - Para datos históricos de precios y volumen
- `/v1/ticker/price` - Para el precio actual
- `/v1/ticker/24hr` - Para estadísticas de las últimas 24 horas

Características de la integración:
- Manejo automático de límites de tasa (rate limiting)
- Paginación para períodos largos de datos
- Manejo de errores y reintentos
- Modo de respaldo con datos sintéticos cuando la API no está disponible

**Nota**: Si no se proporciona una clave API válida, el sistema generará automáticamente datos sintéticos para desarrollo y pruebas.
