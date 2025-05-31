# Sistema de Minería de Datos para Trading Práctico en el Criptomercado (Bitcoin)

Este proyecto implementa un sistema de minería de datos para el criptomercado, enfocado en Bitcoin (BTC), que genera alertas de trading y optimiza decisiones de inversión mediante detección de anomalías y reglas de asociación.

## Descripción

El sistema analiza precios históricos de Bitcoin para:

1. Detectar anomalías y movimientos extremos
2. Agrupar días de trading con características similares (clustering)
3. Descubrir reglas de asociación entre indicadores técnicos y movimientos futuros
4. Evaluar la rentabilidad de estrategias basadas en estas señales

La aplicación es modular, implementada en Python, con una interfaz gráfica para visualizar resultados.

Utiliza la API de Basescan.org para datos históricos y en tiempo real de Bitcoin. También permite operar con datos sintéticos para desarrollo y pruebas.

## Estructura del Proyecto

```
.
├── config.py                    # Configuración global
├── run.py                       # Punto de entrada
├── main.py                      # Script principal
├── test_basescan_api.py         # Script para diagnosticar API
├── requirements.txt             # Dependencias
├── datos/                       # Datos y modelos
├── logs/                        # Archivos de registro
├── src/                         # Código fuente
│   ├── extraccion_datos/        # Extracción de datos
│   ├── preprocesamiento/        # Limpieza y preparación de datos
│   ├── modelos/                 # Algoritmos de minería
│   │   └── clustering.py        # Algoritmo de clustering
│   ├── evaluacion/              # Evaluación de modelos
│   ├── interfaz/                # Interfaz gráfica
│   └── utils/                   # Utilidades
└── tests/                       # Pruebas unitarias
```

## Características

- **Detección de Anomalías**: Identifica movimientos inusuales en retornos o volumen
- **Clustering**: Agrupa días de trading similares
- **Reglas de Asociación**: Encuentra patrones entre indicadores y movimientos
- **Evaluación de Estrategias**: Compara rentabilidad de estrategias
- **Interfaz Gráfica**: Visualiza precios, clusters, anomalías y reglas
- **Modularidad**: Diseño para fácil mantenimiento
- **Resistencia a Fallos**: Manejo de errores y uso de datos sintéticos
- **Diagnóstico API**: Herramientas para verificar conectividad

## Implementaciones

- **Manejo de Errores API**: Fallback a datos alternativos si Basescan.org falla
- **Modo sin API**: Opción `--sin-api` para datos sintéticos
- **Diagnóstico API**: Script para probar conectividad
- **Organización**: Estructura modular del código

## Requisitos

- Python 3.9+
- Dependencias en `requirements.txt`

## Instalación

1. Clonar o descargar el repositorio
2. Crear entorno virtual:
   ```
   python -m venv venv
   ```
3. Activar entorno:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
5. Configurar `config.py` (API_KEY para datos externos)

## Uso

Ejecutar:
```
python run.py
```

Opciones:
- `--modo completo`: Flujo completo (predeterminado)
- `--modo extraccion`: Solo extrae datos
- `--modo preprocesamiento`: Solo preprocesa datos
- `--modo modelado`: Solo entrena modelos
- `--modo evaluacion`: Solo evalúa modelos
- `--modo interfaz`: Solo muestra interfaz
- `--modo test-api`: Diagnostica API
- `--modo limpiar`: Limpia caché
- `--sin-api`: Usa datos sintéticos
- `--forzar-extraccion`: Fuerza extracción
- `--debug`: Muestra depuración
- `--limpiar-cache`: Limpia caché

Ejemplos:
```bash
# Flujo completo con datos sintéticos
python run.py --modo completo --sin-api

# Probar API
python run.py --modo test-api

# Mostrar interfaz
python run.py --modo interfaz

# Forzar extracción
python run.py --modo extraccion --forzar-extraccion

# Modo debug
python run.py --debug
```

## Metodología

Sigue CRISP-DM:
1. **Comprensión del Negocio**: Trading de criptomonedas
2. **Comprensión de los Datos**: Precios y volúmenes de BTC
3. **Preparación de los Datos**: Limpieza e ingeniería
4. **Modelado**: Clustering, anomalías, reglas
5. **Evaluación**: Métricas y backtesting
6. **Despliegue**: Interfaz y alertas

## Métricas de Evaluación

- **Clustering**: Silhouette Score (>0.5)
- **Anomalías**: F1-score (>0.85), precisión, recall
- **Reglas de Asociación**: Confianza (>0.7), lift (>1.2)
- **Estrategias**: Rentabilidad vs. buy-and-hold, máximo drawdown

## Contribuciones

Bienvenidas. Abra un issue para cambios importantes.

## Licencia

Licencia MIT (ver archivo LICENSE).

## Contacto

Autor - Mario Díaz Ramos

## Integración con Basescan.org API

Configuración:
1. Obtener clave API en [Basescan.org](https://basescan.org)
2. Configurar en `config.py`:
   ```python
   API_KEY = "SU_CLAVE_API_AQUI"
   ```

### Diagnóstico de API

Ejecutar:
```bash
python run.py --modo test-api
```

Verifica:
- Conexión a API
- Validez de clave
- Funcionamiento de endpoints
- Disponibilidad de documentación

Resultados en `api_test.log`.

### Modo sin API

Ejecutar:
```bash
python run.py --sin-api
```

Genera datos sintéticos para pruebas.

## Estructura de Módulos

### Extractor de Datos
- Reintentos y espera exponencial
- Manejo de errores HTTP
- Fallback a CoinGecko
- Generación de datos sintéticos

### Detector de Anomalías
- Manejo de tipos de datos
- Respaldo ante errores

### Minería de Reglas
- Manejo de errores
- Documentación

## Conclusiones

Sistema modular para análisis de criptomonedas, con énfasis en anomalías y reglas de asociación. Garantiza:
1. Resistencia a fallos
2. Operación con o sin API
3. Código organizado

Extensible para nuevos algoritmos, indicadores y fuentes de datos.

El sistema usa los endpoints de Basescan.org:
- `/v1/klines`: Datos históricos
- `/v1/ticker/price`: Precio actual
- `/v1/ticker/24hr`: Estadísticas 24h

Características:
- Manejo de límites de tasa
- Paginación para datos extensos
- Reintentos y manejo de errores
- Modo de respaldo con datos sintéticos

**Nota**: Sin clave API válida, se generan datos sintéticos.