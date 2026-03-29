# Plan de Desarrollo - Script de Predicciones MLT

> **NOTA:** Este documento ha sido **SUPERADO** por `PLAN_MATEMATICO_MLT.md` (v3) y `PLAN_PASO_A_PASO.md`. Se mantiene como referencia historica del plan original. Para el plan vigente, consultar `PLAN_MATEMATICO_MLT.md`.

## Objetivo

Crear un script en Python que genere predicciones del MLT a partir de resultados históricos contenidos en un archivo CSV.

---

## Fases del Proyecto

### Fase 1: Servicio de Obtención de Datos

Crear un servicio que se encargue de obtener el archivo CSV con los resultados anteriores y los datos necesarios para realizar la predicción.

**Tareas:**

1. **Definir la fuente de datos**: Determinar de dónde se obtendrá el archivo CSV (ruta local, URL remota, API, etc.).
2. **Crear el módulo `data_service.py`**: Servicio encargado de:
   - Descargar o leer el archivo CSV con los resultados históricos.
   - Validar la estructura y calidad de los datos (columnas esperadas, valores nulos, tipos de dato).
   - Retornar un DataFrame limpio y listo para el análisis.
3. **Manejo de errores**: Implementar control de excepciones para casos como archivo no encontrado, formato inválido o datos corruptos.

---

### Fase 2: Exploración y Preparación de Datos

Analizar los datos históricos para entender su estructura y prepararlos para el modelo.

**Tareas:**

1. **Análisis exploratorio (EDA)**: Estadísticas descriptivas, distribuciones, frecuencias y patrones.
2. **Limpieza de datos**: Tratamiento de valores faltantes, duplicados y outliers.
3. **Ingeniería de features**: Crear variables derivadas que puedan mejorar la capacidad predictiva del modelo (tendencias, promedios móviles, frecuencias acumuladas, etc.).

---

### Fase 3: Modelado y Predicción

Entrenar un modelo de Machine Learning para generar las predicciones.

**Tareas:**

1. **Selección del modelo**: Evaluar algoritmos candidatos (Random Forest, Gradient Boosting, redes neuronales, etc.) usando scikit-learn.
2. **Entrenamiento**: Dividir los datos en conjuntos de entrenamiento y prueba; entrenar el modelo seleccionado.
3. **Evaluación**: Medir el rendimiento del modelo con métricas apropiadas (accuracy, RMSE, MAE, etc.).
4. **Generación de predicciones**: Ejecutar el modelo con los datos más recientes para producir las predicciones del MLT.

---

### Fase 4: Salida de Resultados

Presentar y almacenar las predicciones generadas.

**Tareas:**

1. **Exportar predicciones**: Guardar los resultados en un archivo CSV o formato configurable.
2. **Visualización**: Generar gráficos de las predicciones usando matplotlib/seaborn.
3. **Logging**: Registrar cada ejecución con fecha, parámetros del modelo y resultados obtenidos.

---

### Fase 5: Integración y Automatización

Empaquetar todo en un script ejecutable y reproducible.

**Tareas:**

1. **Script principal `predict.py`**: Orquestar todo el flujo (obtención de datos → preparación → predicción → salida).
2. **Configuración**: Archivo de configuración (`.env` o `config.yaml`) para parámetros como rutas de archivos, hiperparámetros del modelo, etc.
3. **Documentación**: Instrucciones de uso en el README del repositorio.

---

## Estructura Propuesta del Proyecto

```
MLT/
├── data/                  # Archivos CSV (excluidos de git)
├── src/
│   ├── data_service.py    # Servicio de obtención y validación de datos
│   ├── preprocessing.py   # Limpieza e ingeniería de features
│   ├── model.py           # Entrenamiento y evaluación del modelo
│   └── predict.py         # Script principal de predicción
├── outputs/               # Predicciones y visualizaciones generadas
├── notebooks/             # Jupyter notebooks para exploración
├── requirements.txt       # Dependencias del proyecto
├── PLAN_PREDICCIONES.md   # Este archivo
└── README.md
```

---

## Stack Tecnológico

| Herramienta    | Uso                                      |
| :------------- | :--------------------------------------- |
| Python 3.x     | Lenguaje principal                       |
| pandas         | Manipulación y análisis de datos         |
| numpy          | Cálculos numéricos                       |
| scikit-learn   | Modelos de Machine Learning              |
| matplotlib     | Visualización de datos                   |
| seaborn        | Visualización estadística                |
| requests       | Obtención de datos desde fuentes remotas |
| JupyterLab     | Exploración interactiva                  |

---

## Próximos Pasos

1. Comenzar con la **Fase 1**: implementar `data_service.py` para obtener y validar el CSV con los resultados históricos.
2. Definir con el equipo la fuente exacta del archivo CSV y las columnas esperadas.
