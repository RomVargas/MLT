# Plan Paso a Paso - Implementacion del Sistema de Prediccion MLT

> **Documento de referencia tecnico:** `PLAN_MATEMATICO_MLT.md` (v2 con correcciones)
> **Entorno de desarrollo:** Jupyter Notebook
> **Convenciones:** Cada celda de codigo incluye comentarios matematicos en LaTeX

---

## Fase 0: Configuracion del Entorno

**Archivo:** `notebooks/00_setup_entorno.ipynb`

| Paso | Tarea | Validacion |
|------|-------|------------|
| 0.1 | Importar todas las librerias del stack | Sin errores de importacion |
| 0.2 | Verificar versiones de cada libreria | Versiones coinciden con `requirements.txt` |
| 0.3 | Detectar y verificar GPU disponible (CUDA/ROCm) | `torch.cuda.is_available()` o `xgboost` GPU |
| 0.4 | Configurar semilla global de reproducibilidad | `RANDOM_STATE = 42` en todos los componentes |
| 0.5 | Configurar matplotlib para notebooks (`%matplotlib inline`) | Graficos se muestran en celdas |
| 0.6 | Verificar entorno Jupyter (version, kernel) | Kernel Python 3.x activo |

**Criterio de exito:** Todas las librerias importadas, GPU detectada (o documentado que se usa CPU), semilla fijada.

---

## Fase 1: Obtencion y Validacion de Datos

**Archivo:** `notebooks/01_data_service.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 1

| Paso | Tarea | Validacion |
|------|-------|------------|
| 1.1 | Definir ruta del CSV con resultados historicos | Archivo accesible |
| 1.2 | Cargar CSV con `pandas.read_csv()` | DataFrame cargado sin errores |
| 1.3 | Validar esquema (columnas: sorteo, fecha, r1..r7) | Columnas y tipos correctos |
| 1.4 | Detectar y reportar valores nulos | `df.isnull().sum()` por columna |
| 1.5 | Validar rango de numeros `[1, 56]` | Ningun valor fuera de rango |
| 1.6 | Detectar sorteos duplicados | 0 duplicados |
| 1.7 | Ordenar cronologicamente | `df.sort_values('fecha')` |
| 1.8 | Generar hash de version de datos | `sha256` registrado |
| 1.9 | Mostrar resumen: shape, dtypes, head, tail | Visualizacion correcta |

**Criterio de exito:** DataFrame limpio, ordenado cronologicamente, con hash de version registrado.

---

## Fase 2: Analisis Exploratorio de Datos (EDA)

**Archivo:** `notebooks/02_eda.ipynb`
**Referencia:** PLAN_MATEMATICO Pasos 2 y 3

| Paso | Tarea | Validacion |
|------|-------|------------|
| 2.1 | Estadisticas descriptivas por posicion (media, std, skewness, kurtosis) | Tabla con LaTeX |
| 2.2 | Frecuencia absoluta y relativa de cada numero [1..56] | Grafico de barras |
| 2.3 | Test chi-cuadrado de uniformidad | p-value reportado |
| 2.4 | Clasificacion hot/cold/overdue (solo descriptivo) | Listas con advertencia de falacia |
| 2.5 | Correlacion de Pearson entre posiciones (heatmap) | Heatmap seaborn |
| 2.6 | Informacion Mutua entre posiciones | Tabla de MI |
| 2.7 | Analisis de pares frecuentes + z-scores | Top 20 pares significativos |
| 2.8 | Media Movil Simple y Exponencial | Graficos de tendencia |
| 2.9 | Autocorrelacion (ACF) por numero | Graficos ACF + decision sobre Markov |
| 2.10 | Tests de independencia secuencial (Runs Test, Ljung-Box) | p-values para decision sobre Cadenas de Markov |
| 2.11 | Convergencia LGN: delta(k,n) vs 1/sqrt(n) | Grafico de convergencia |

**Criterio de exito:** Reporte EDA completo. Decision documentada sobre si hay patrones explotables o si los datos son consistentes con aleatoriedad pura.

---

## Fase 3: Ingenieria de Features

**Archivo:** `notebooks/03_feature_engineering.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 4

| Paso | Tarea | Validacion |
|------|-------|------------|
| 3.1 | Generar features por sorteo (suma, media, rango, varianza, pares, impares, consecutivos, decenas, primos, suma_digitos, ratio_alto_bajo, gap_max, gap_medio) | DataFrame ampliado |
| 3.2 | Generar features temporales con ventana deslizante (freq_reciente, tendencia, rezago, ciclo_medio, aceleracion) | Sin fuga temporal (solo datos pasados) |
| 3.3 | Generar features combinatorios (entropia Shannon, distancia Mahalanobis) | Valores calculados |
| 3.4 | **Validacion de fuga temporal**: verificar que cada feature(t) solo usa datos de sorteos 1..t-1 | Test de causalidad aprobado |
| 3.5 | **Validacion de estabilidad**: CV inter-ventana < 0.5 para cada feature | Features inestables descartados |
| 3.6 | **Seleccion formal (MI)**: mutual_information_classif para cada feature | Ranking de importancia |
| 3.7 | **Seleccion formal (RFE)**: eliminacion recursiva con RF base | Top features seleccionados |
| 3.8 | **Validacion Boruta-like**: comparar features reales vs permutados | Features no informativos descartados |
| 3.9 | Normalizar features seleccionados (StandardScaler / MinMaxScaler) | Pipeline sklearn creado |
| 3.10 | Documentar features finales seleccionados | Lista definitiva con justificacion |

**Criterio de exito:** Conjunto final de 5-8 features validados, sin fuga temporal, estables y superiores al ruido aleatorio.

---

## Fase 4: Baseline y Modelos

**Archivo:** `notebooks/04_modelos.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 5

| Paso | Tarea | Validacion |
|------|-------|------------|
| 4.1 | Implementar baseline uniforme aleatorio | E[aciertos] = 0.643 por sorteo |
| 4.2 | Implementar Frecuencias Bayesianas (prior Beta) | Posterior calculada, IC 95% |
| 4.3 | Evaluar Bayesiano vs baseline con test binomial | p-value reportado |
| 4.4 | Implementar Random Forest Classifier (`class_weight='balanced'`) | Modelo entrenado |
| 4.5 | Hyperparameter tuning RF con Optuna + TimeSeriesSplit | Mejores hiperparametros |
| 4.6 | Evaluar RF vs baseline | p-value reportado |
| 4.7 | Implementar Gradient Boosting (XGBoost, `class_weight='balanced'`, GPU) | Modelo con GPU |
| 4.8 | Hyperparameter tuning GB con Optuna + TimeSeriesSplit | Mejores hiperparametros |
| 4.9 | Evaluar GB vs baseline | p-value reportado |
| 4.10 | **Condicional:** Si tests de independencia (Fase 2.10) lo justifican, implementar Cadenas de Markov | Modelo solo si p < 0.05 |
| 4.11 | Calibracion de probabilidades (Platt Scaling / Isotonic) | Reliability diagram |
| 4.12 | Comparacion de todos los modelos con metricas primarias | Tabla comparativa |

**Criterio de exito:** Al menos un modelo supera al baseline con p-value < 0.05 en walk-forward, O se documenta que ningun modelo lo supera.

---

## Fase 5: Evaluacion Walk-Forward

**Archivo:** `notebooks/05_evaluacion.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 5.7

| Paso | Tarea | Validacion |
|------|-------|------------|
| 5.1 | Implementar walk-forward validation completo | Evaluacion en cada sorteo |
| 5.2 | Calcular metricas primarias: Hit Rate, Distribucion de Aciertos, ROI simulado | Tablas y graficos |
| 5.3 | Calcular metricas secundarias: Brier Score, Log-Loss | Valores reportados |
| 5.4 | Comparacion contra 10,000 simulaciones aleatorias | Percentil del modelo |
| 5.5 | Test binomial formal vs baseline | p-value final |
| 5.6 | **Decision:** modo PREDICTIVO o DESCRIPTIVO | Documentado con evidencia |

**Criterio de exito:** Decision fundamentada sobre si el sistema opera en modo predictivo o descriptivo.

---

## Fase 6: Ensamble y Prediccion Final

**Archivo:** `notebooks/06_prediccion.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 6

| Paso | Tarea | Validacion |
|------|-------|------------|
| 6.1 | **Si modo PREDICTIVO:** Construir ensamble ponderado (pesos por Brier Score / Hit Rate) | P_final(k) para cada numero |
| 6.2 | Seleccion Top-K y muestreo ponderado | Combinacion(es) generada(s) |
| 6.3 | Aplicar restricciones combinatorias (rango, par/impar, decenas) | Combinaciones validas |
| 6.4 | Simulacion Monte Carlo (robustez) | Intervalos de confianza |
| 6.5 | **Si modo DESCRIPTIVO:** Generar reporte de patrones, frecuencias, tendencias | Reporte analitico sin prediccion numerica |

**Criterio de exito:** Predicciones generadas con intervalos de confianza, o reporte descriptivo completo.

---

## Fase 7: Exportacion y Logging

**Archivo:** `notebooks/07_exportar.ipynb`
**Referencia:** PLAN_MATEMATICO Paso 7

| Paso | Tarea | Validacion |
|------|-------|------------|
| 7.1 | Exportar predicciones a CSV | Archivo generado |
| 7.2 | Generar graficos finales (distribuciones, heatmaps, tendencias) | PNGs en `outputs/` |
| 7.3 | Registrar log de ejecucion (timestamp, version datos, hiperparametros, metricas) | Log persistido |
| 7.4 | Guardar modelos entrenados con joblib | Archivos `.pkl` en `models/` |

---

## Estructura de Archivos del Proyecto

```
MLT/
├── data/                           # CSVs historicos (excluidos de git)
├── models/                         # Modelos serializados (.pkl)
├── notebooks/
│   ├── 00_setup_entorno.ipynb      # <-- EMPEZAMOS AQUI
│   ├── 01_data_service.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelos.ipynb
│   ├── 05_evaluacion.ipynb
│   ├── 06_prediccion.ipynb
│   └── 07_exportar.ipynb
├── outputs/                        # Predicciones, graficos, logs
├── src/                            # Modulos reutilizables (extraidos de notebooks)
│   ├── data_service.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── predict.py
├── requirements.txt
├── PLAN_MATEMATICO_MLT.md
├── PLAN_PASO_A_PASO.md             # Este archivo
├── PLAN_PREDICCIONES.md
└── README.md
```

---

## Dependencias Adicionales (agregar a requirements.txt)

```
xgboost          # Gradient Boosting con soporte GPU
optuna           # Hyperparameter tuning bayesiano
```

---

## Reglas de Desarrollo

1. **Cada celda de Jupyter incluye comentarios LaTeX** explicando las formulas matematicas aplicadas.
2. **Cada fase genera un checkpoint guardable** (DataFrame intermedio, modelo serializado, etc.).
3. **random_state=42** en todo componente estocastico.
4. **Ningun feature usa datos futuros** (validacion obligatoria en Fase 3).
5. **Todo modelo se compara contra el baseline** antes de aceptarse.
6. **Si ningun modelo supera al azar**, el sistema opera en modo descriptivo con total honestidad.
