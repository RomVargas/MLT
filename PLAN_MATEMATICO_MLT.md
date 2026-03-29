# Plan Matematico y Tecnico - Prediccion MLT

## Introduccion

Este documento detalla el plan tecnico y matematico para construir un sistema de prediccion de resultados del MLT (Melate) basado en datos historicos. Se describen las formulas matematicas, procesos estadisticos, algoritmos de Machine Learning y la justificacion de cada libreria seleccionada.

> **Nota importante:** Los sorteos de loteria son eventos estocasticos independientes. Ningun modelo puede garantizar predicciones exactas. El objetivo de este proyecto es aplicar tecnicas estadisticas y de ML para identificar patrones, tendencias y distribuciones que permitan generar combinaciones con mayor fundamento analitico que la seleccion aleatoria.

> **Principio rector:** Todo modelo debe compararse obligatoriamente contra un **baseline uniforme** (seleccion aleatoria). Si ningun modelo supera al baseline de forma estadisticamente significativa en backtesting temporal, el sistema adoptara un enfoque **descriptivo** (analisis de patrones y frecuencias) en lugar de predictivo. La honestidad estadistica es prioritaria.

---

## Paso 1: Obtencion del Archivo CSV con Resultados Historicos

### 1.1 Descripcion

El primer paso es obtener y cargar el archivo CSV que contiene los resultados historicos de los sorteos del MLT. Este archivo es la base de todo el analisis posterior.

### 1.2 Estructura Esperada del CSV

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| `sorteo` | int | Numero identificador del sorteo |
| `fecha` | date | Fecha del sorteo (YYYY-MM-DD) |
| `r1` | int | Primer numero sorteado |
| `r2` | int | Segundo numero sorteado |
| `r3` | int | Tercer numero sorteado |
| `r4` | int | Cuarto numero sorteado |
| `r5` | int | Quinto numero sorteado |
| `r6` | int | Sexto numero sorteado |
| `r7` | int | Numero adicional (si aplica) |

### 1.3 Proceso de Carga y Validacion

1. **Lectura del CSV** con `pandas.read_csv()`.
2. **Validacion de esquema**: verificar que las columnas esperadas existen y tienen el tipo correcto.
3. **Deteccion de valores nulos**: `df.isnull().sum()` por columna.
4. **Validacion de rango**: todos los numeros deben estar en el intervalo `[1, N]` donde `N` es el numero maximo del juego (ej. 56 para Melate).
5. **Deteccion de duplicados**: verificar que no haya sorteos duplicados.
6. **Ordenamiento cronologico**: ordenar por fecha ascendente para analisis de series temporales.

### 1.4 Libreria: pandas

**Por que pandas es la opcion optima:**

- **Rendimiento**: Construida sobre NumPy (C/Fortran), ofrece operaciones vectorizadas que son ordenes de magnitud mas rapidas que loops nativos de Python.
- **API de lectura**: `read_csv()` soporta parsing automatico de fechas, tipos, encoding, separadores y manejo de valores faltantes.
- **DataFrame**: Estructura tabular con indices, que permite seleccion, filtrado, agrupacion y transformacion de datos con sintaxis concisa.
- **Integracion**: Se integra nativamente con NumPy, scikit-learn, matplotlib y seaborn.
- **Alternativas descartadas**:
  - `csv` (stdlib): No soporta operaciones vectorizadas ni tipos de datos avanzados.
  - `polars`: Mas rapido en datasets muy grandes (>1GB), pero para nuestro volumen (~5,000 filas) no justifica la curva de aprendizaje adicional.
  - `dask`: Diseñado para datasets que no caben en memoria. Innecesario para nuestro caso.

---

## Paso 2: Analisis Exploratorio de Datos (EDA)

### 2.1 Estadisticas Descriptivas Fundamentales

#### 2.1.1 Media Aritmetica

La media de cada posicion nos indica el centro de la distribucion:

```
         n
        SUM x_i
        i=1
mu = -----------
          n
```

Donde `x_i` es el numero en la posicion `j` del sorteo `i`, y `n` es el total de sorteos.

#### 2.1.2 Varianza y Desviacion Estandar

Miden la dispersion de los numeros alrededor de la media:

```
          n
         SUM (x_i - mu)^2
         i=1
s^2 = ---------------------
            n - 1


s = sqrt(s^2)
```

La desviacion estandar (`s`) nos dice que tan "repartidos" estan los numeros. Una desviacion alta indica que los numeros se distribuyen ampliamente en el rango; una baja, que se concentran.

#### 2.1.3 Asimetria (Skewness)

Mide si la distribucion esta sesgada hacia la izquierda o derecha:

```
              n
       (1/n) SUM (x_i - mu)^3
             i=1
g1 = --------------------------
             s^3
```

- `g1 > 0`: sesgo a la derecha (cola larga hacia numeros altos)
- `g1 < 0`: sesgo a la izquierda (cola larga hacia numeros bajos)
- `g1 = 0`: distribucion simetrica

#### 2.1.4 Curtosis (Kurtosis)

Mide el "peso" de las colas de la distribucion:

```
              n
       (1/n) SUM (x_i - mu)^4
             i=1
g2 = -------------------------- - 3
             s^4
```

- `g2 > 0` (leptocurtica): colas pesadas, mas valores extremos de lo esperado.
- `g2 < 0` (platicurtica): colas ligeras, menos valores extremos.
- `g2 = 0` (mesocurtica): similar a una distribucion normal.

### 2.2 Analisis de Frecuencias

#### 2.2.1 Frecuencia Absoluta

Para cada numero `k` en el rango `[1, N]`:

```
f(k) = cantidad de sorteos donde k aparece como uno de los numeros ganadores
```

#### 2.2.2 Frecuencia Relativa

```
                f(k)
f_rel(k) = -----------
            n * m
```

Donde `n` es el total de sorteos y `m` es la cantidad de numeros por sorteo (ej. 6).

En un juego perfectamente aleatorio con distribucion uniforme, la frecuencia relativa esperada para cada numero es:

```
                    m
f_rel_esperada = -------
                   N
```

Para Melate (6 de 56): `f_rel_esperada = 6/56 = 0.1071` (10.71%).

#### 2.2.3 Test Chi-Cuadrado de Bondad de Ajuste

Para verificar si las frecuencias observadas se desvian significativamente de una distribucion uniforme:

```
         N
        SUM (O_k - E_k)^2
        k=1
X^2 = ---------------------
             E_k
```

Donde:
- `O_k` = frecuencia observada del numero `k`
- `E_k` = frecuencia esperada = `(n * m) / N`
- Grados de libertad: `df = N - 1`

Si `X^2 > X^2_critico(alpha, df)` entonces rechazamos la hipotesis nula de uniformidad, lo que sugiere que ciertos numeros aparecen mas/menos de lo esperado.

### 2.3 Clasificacion de Numeros

> **Advertencia sobre la Falacia del Jugador:** Si los sorteos son eventos independientes (como se espera en loterias modernas), las frecuencias pasadas **no predicen** resultados futuros. Que un numero haya aparecido mucho (caliente) o poco (frio) no altera su probabilidad en el proximo sorteo. Esta clasificacion es util para el **analisis descriptivo** (entender la distribucion historica), pero su valor predictivo debe ser **validado formalmente** con el test chi-cuadrado (Paso 2.2.3) antes de incluir estos features en modelos predictivos. Solo si el test rechaza la hipotesis de uniformidad se justifica usar frecuencias como features.

#### 2.3.1 Numeros Calientes (Hot Numbers)

```
Hot(k) = True   si   f(k) > mu_f + z * sigma_f
```

Donde:
- `mu_f` = media de las frecuencias de todos los numeros
- `sigma_f` = desviacion estandar de las frecuencias
- `z` = umbral (tipicamente 1.0 para top ~16%)

#### 2.3.2 Numeros Frios (Cold Numbers)

```
Cold(k) = True   si   f(k) < mu_f - z * sigma_f
```

#### 2.3.3 Numeros Rezagados (Overdue Numbers)

Sea `d(k)` el numero de sorteos desde la ultima aparicion de `k`:

```
Overdue(k) = True   si   d(k) > mu_d + z * sigma_d
```

Donde `mu_d` y `sigma_d` son la media y desviacion estandar de los rezagos de todos los numeros.

### 2.4 Libreria: NumPy

**Por que NumPy es la opcion optima:**

- **Fundamento computacional**: Todas las operaciones matematicas (media, varianza, etc.) se ejecutan como operaciones vectorizadas en C, lo que es 10-100x mas rapido que loops de Python.
- **Precision numerica**: Usa tipos `float64` por defecto, evitando errores de punto flotante en calculos estadisticos.
- **Broadcasting**: Permite operaciones entre arrays de diferentes dimensiones sin copiar datos.
- **Base del ecosistema**: pandas, scikit-learn, scipy y matplotlib dependen de NumPy internamente. Usar la misma base evita conversiones costosas.

---

## Paso 3: Analisis de Patrones Avanzado

### 3.1 Ley de los Grandes Numeros (LGN)

La LGN establece que a medida que el numero de ensayos (`n`) tiende a infinito, la media muestral converge a la media teorica:

```
lim    X_barra_n = mu
n->inf
```

Formalmente (Ley Debil):

```
Para todo epsilon > 0:

lim  P(|X_barra_n - mu| >= epsilon) = 0
n->inf
```

**Aplicacion en MLT:** Con suficientes sorteos, la frecuencia relativa de cada numero debe converger a `m/N`. Las desviaciones persistentes pueden indicar:
- **Muestra insuficiente**: Necesitamos mas datos.
- **Anomalias estadisticas reales**: Improbable en loterias modernas pero vale la pena documentar.

Medimos la tasa de convergencia con:

```
delta(k, n) = |f_rel(k, n) - m/N|
```

Si `delta(k, n)` decrece a tasa `O(1/sqrt(n))`, el comportamiento es consistente con la LGN.

### 3.2 Teorema del Limite Central (TLC)

La suma/media de variables aleatorias independientes tiende a una distribucion normal:

```
Z = (X_barra_n - mu) / (sigma / sqrt(n))  ~  N(0, 1)
```

**Aplicacion:** Nos permite construir intervalos de confianza para las frecuencias esperadas:

```
IC(1-alpha) = [mu_f - z_(alpha/2) * sigma_f/sqrt(n),  mu_f + z_(alpha/2) * sigma_f/sqrt(n)]
```

Numeros cuya frecuencia cae fuera de este intervalo son estadisticamente anomalos.

### 3.3 Analisis de Correlacion entre Posiciones

#### 3.3.1 Coeficiente de Correlacion de Pearson

Para evaluar si existe relacion lineal entre las posiciones `j` y `k`:

```
                   n
                  SUM (x_ij - mu_j)(x_ik - mu_k)
                  i=1
r(j,k) = -----------------------------------------
           sqrt(SUM(x_ij - mu_j)^2) * sqrt(SUM(x_ik - mu_k)^2)
```

- `r = 1`: correlacion positiva perfecta
- `r = -1`: correlacion negativa perfecta
- `r = 0`: sin correlacion lineal

#### 3.3.2 Informacion Mutua (Mutual Information)

Para capturar dependencias no lineales entre posiciones:

```
                         p(x, y)
I(X;Y) = SUM SUM p(x,y) log -----------
          x    y              p(x) p(y)
```

Donde `p(x,y)` es la probabilidad conjunta y `p(x)`, `p(y)` son las marginales. Si `I(X;Y) > 0`, existe alguna dependencia (lineal o no) entre las posiciones.

### 3.4 Analisis de Pares y Tripletas Frecuentes

#### 3.4.1 Frecuencia de Pares

Para cada par `(a, b)` donde `a < b`:

```
f_par(a,b) = numero de sorteos donde AMBOS a y b aparecen
```

Frecuencia esperada bajo independencia:

```
                         C(N-2, m-2)
E[f_par(a,b)] = n * -------------------
                         C(N, m)
```

Donde `C(n,k)` es el coeficiente binomial: `C(n,k) = n! / (k!(n-k)!)`

#### 3.4.2 Z-Score de Pares

```
                    f_par(a,b) - E[f_par(a,b)]
z_par(a,b) = -----------------------------------
               sqrt(E[f_par(a,b)] * (1 - p_par))
```

Donde `p_par = C(N-2, m-2) / C(N, m)`.

Pares con `|z_par| > 2` son estadisticamente significativos al 95%.

### 3.5 Analisis de Series Temporales

#### 3.5.1 Media Movil Simple (SMA)

Para suavizar las fluctuaciones en las frecuencias a lo largo del tiempo:

```
                  1    t
SMA(k, t, w) = ----- SUM f(k, i)
                 w   i=t-w+1
```

Donde `w` es la ventana (ej. ultimos 50 sorteos).

#### 3.5.2 Media Movil Exponencial (EMA)

Da mas peso a las observaciones recientes:

```
EMA(t) = alpha * x(t) + (1 - alpha) * EMA(t-1)
```

Donde el factor de suavizado es:

```
alpha = 2 / (w + 1)
```

La EMA es preferible a la SMA porque captura cambios de tendencia mas rapidamente.

#### 3.5.3 Autocorrelacion

Mide la correlacion de la serie de frecuencias consigo misma desplazada `lag` periodos:

```
                     n-lag
                    SUM (x_t - mu)(x_{t+lag} - mu)
                    t=1
ACF(lag) = -------------------------------------------
                     n
                    SUM (x_t - mu)^2
                    t=1
```

Si `ACF(lag) != 0` para algun `lag`, existe un patron ciclico que podemos explotar.

### 3.6 Librerias: SciPy y seaborn

**Por que SciPy:**

- **scipy.stats**: Implementa tests estadisticos (chi-cuadrado, Kolmogorov-Smirnov, etc.) con p-values exactos.
- **scipy.special**: Funciones para coeficientes binomiales, gamma, etc.
- **Precision**: Usa algoritmos numericamente estables para evitar overflow/underflow en factoriales grandes.
- **Alternativas descartadas**:
  - `statsmodels`: Mas orientado a modelos econometricos. Excesivo para nuestros tests basicos.
  - Implementacion manual: Propensa a errores numericos en factoriales y distribuciones.

**Por que seaborn:**

- **Visualizacion estadistica**: Diseñado especificamente para graficos estadisticos (heatmaps de correlacion, distribuciones, boxplots).
- **Integracion con pandas**: Acepta DataFrames directamente, sin conversion.
- **Estetica superior**: Produce graficos publicables con configuracion minima.
- **Construido sobre matplotlib**: Hereda toda la flexibilidad de matplotlib pero con una API de alto nivel.

---

## Paso 4: Ingenieria de Features (Variables Derivadas)

### 4.1 Features por Sorteo

Para cada sorteo generamos las siguientes variables:

| Feature | Formula | Descripcion |
|---------|---------|-------------|
| `suma` | `SUM(r1..r6)` | Suma de los 6 numeros |
| `media` | `suma / 6` | Media aritmetica |
| `rango` | `r6 - r1` | Diferencia entre el mayor y menor numero |
| `varianza` | `SUM(ri - media)^2 / 5` | Dispersion intra-sorteo |
| `pares` | `count(ri % 2 == 0)` | Cantidad de numeros pares |
| `impares` | `6 - pares` | Cantidad de numeros impares |
| `consecutivos` | `count(r_{i+1} - r_i == 1)` | Pares consecutivos |
| `decenas` | `count(unique(ri // 10))` | Diversidad de decenas representadas |
| `primos` | `count(ri in PRIMOS)` | Cantidad de numeros primos |
| `suma_digitos` | `SUM(sum_digitos(ri))` | Suma de todos los digitos |
| `ratio_alto_bajo` | `count(ri > N/2) / 6` | Proporcion de numeros "altos" |
| `gap_max` | `max(r_{i+1} - r_i)` | Mayor brecha entre numeros consecutivos (ordenados) |
| `gap_medio` | `mean(r_{i+1} - r_i)` | Brecha promedio |

### 4.2 Features Temporales (Ventana Deslizante)

Para cada numero `k` y una ventana de `w` sorteos previos:

| Feature | Formula | Descripcion |
|---------|---------|-------------|
| `freq_reciente` | `f(k, ultimos w) / w` | Frecuencia en ventana reciente |
| `tendencia` | `EMA(freq, w) - SMA(freq, w)` | Diferencia EMA vs SMA (momentum) |
| `rezago` | `sorteos_desde_ultima_aparicion(k)` | Cuantos sorteos sin aparecer |
| `ciclo_medio` | `mean(intervalos_entre_apariciones(k))` | Intervalo promedio entre apariciones |
| `aceleracion` | `freq(w/2 reciente) - freq(w/2 anterior)` | Cambio en tendencia |

### 4.3 Features Combinatorias

#### 4.3.1 Entropia de Shannon del Sorteo

Mide la "sorpresa" o diversidad de un sorteo:

```
H = -SUM p_i * log2(p_i)
```

Donde `p_i = r_i / SUM(r_j)`. Sorteos con numeros muy dispersos tienen mayor entropia.

> **Nota:** Esta definicion de `p_i` es una aproximacion heuristica para asignar pesos a los numeros dentro de un sorteo. No representa una distribucion de probabilidad formal del juego, sino una medida relativa de la dispersion de valores.

#### 4.3.2 Distancia de Mahalanobis

Mide que tan "atipico" es un sorteo respecto a la distribucion historica:

```
D_M = sqrt((x - mu)^T * S^(-1) * (x - mu))
```

Donde:
- `x` = vector del sorteo [r1, r2, ..., r6]
- `mu` = vector de medias historicas por posicion
- `S` = matriz de covarianza de los sorteos historicos

Sorteos con `D_M` alto son atipicos. Esto nos ayuda a definir que combinaciones son "razonables".

### 4.4 Validacion de Features contra Fuga Temporal

Antes de usar cualquier feature en el modelo, debemos verificar que **no contenga informacion del futuro** (data leakage temporal):

1. **Regla estricta**: Cada feature del sorteo `t` solo puede usar datos de sorteos `1..t-1`.
2. **Validacion de estabilidad**: Calcular cada feature en ventanas de 200 sorteos consecutivos y medir su varianza inter-ventana:

```
                      K
Estabilidad(f) = 1 - CV(f) = 1 - (std(f_1..f_K) / mean(f_1..f_K))
```

Donde `f_k` es el valor del feature en la ventana `k`. Features con `Estabilidad < 0.5` son inestables y deben descartarse.

3. **Test de causalidad temporal**: Para cada feature, comparar su correlacion con el target en datos pasados vs. datos futuros. Si la correlacion futura es significativamente mayor, el feature tiene fuga temporal.

### 4.5 Seleccion Formal de Features (Prevencion de Overfitting)

> **Correccion v2:** El plan original define ~18+ features para un dataset de ~5,000 sorteos. Esto genera un riesgo serio de overfitting, especialmente en modelos como Gradient Boosting que pueden memorizar ruido. Se agrega esta fase obligatoria de seleccion.

#### 4.5.1 Filtrado por Informacion Mutua

Calcular la informacion mutua entre cada feature y el target (aparicion de cada numero):

```
MI(feature_j, target_k) = mutual_information_classif(X[:, j], y_k)
```

Features con `MI < umbral` se descartan. El umbral se determina empiricamente comparando contra features aleatorios (permutation importance).

#### 4.5.2 Eliminacion Recursiva de Features (RFE)

Usando el Random Forest como estimador base:

```
Para i = num_features hasta min_features:
    1. Entrenar RF con features actuales
    2. Calcular importancia de cada feature
    3. Eliminar el feature menos importante
    4. Evaluar rendimiento con walk-forward
    5. Registrar score

Seleccionar el subconjunto con mejor score en walk-forward
```

#### 4.5.3 Regla de Dimensionalidad

Limitar el numero maximo de features a:

```
max_features = sqrt(n_sorteos_entrenamiento)
```

Para ~5,000 sorteos: `max_features ~ 70`. Pero en la practica, limitaremos a **los top 5-8 features mas informativos** para mantener el modelo interpretable y robusto.

#### 4.5.4 Validacion con Features Aleatorios (Boruta-like)

Para cada feature real, generar una version permutada aleatoriamente. Si la importancia del feature real no supera significativamente la del feature aleatorio, descartarlo:

```
Para cada feature_j:
    feature_j_aleatorio = np.random.permutation(feature_j)
    Si Imp(feature_j) <= max(Imp(features_aleatorios)):
        -> Descartar feature_j (no aporta mas que ruido)
```

### 4.6 Libreria: scikit-learn (sklearn)

**Por que scikit-learn es la opcion optima para ingenieria de features:**

- **StandardScaler**: Normalizacion z-score `z = (x - mu) / sigma` para que todos los features tengan la misma escala.
- **MinMaxScaler**: Normalizacion al rango `[0, 1]`: `x_norm = (x - x_min) / (x_max - x_min)`.
- **PolynomialFeatures**: Genera automaticamente terminos polinomiales e interacciones.
- **Pipeline**: Encadena transformaciones de forma reproducible.
- **Alternativas descartadas**:
  - `feature-engine`: Mas especifico pero menos documentado y con comunidad mas pequeña.
  - Implementacion manual: Propensa a errores y no ofrece la integracion con pipelines de sklearn.

---

## Paso 5: Modelos de Machine Learning para Prediccion

### 5.0 Manejo del Desbalance de Clases

> **Correccion v2:** Cada numero aparece en ~10.7% de los sorteos (clase positiva) y no aparece en ~89.3% (clase negativa). Este desbalance no se abordaba en la version anterior. Sin tratamiento, los modelos tienden a predecir "no aparece" para todos los numeros.

**Estrategias de manejo (en orden de prioridad):**

1. **Pesos de clase balanceados**: Configurar `class_weight='balanced'` en RF y GB:

```
w_clase = n_total / (n_clases * n_muestras_clase)

w_positiva = n / (2 * f(k))      ~= 4.67  (para frecuencia tipica)
w_negativa = n / (2 * (n - f(k))) ~= 0.56
```

2. **Threshold tuning**: En lugar de usar 0.5 como umbral de decision, optimizar el threshold que maximiza el Hit Rate:

```
threshold_optimo = argmax_t [Hit_Rate(P > t)]
```

3. **Metricas sensibles al desbalance**: Usar Brier Score y Log-Loss (ya incluidos en 5.6.2) que penalizan naturalmente las probabilidades mal calibradas, en lugar de accuracy que seria engañosa (~89% prediciendo siempre "no aparece").

### 5.0.1 Reformulacion del Problema: Clasificacion Multi-etiqueta Combinatoria

**Cambio critico respecto al enfoque original:** En lugar de tratar la prediccion como un problema de **regresion** (predecir numeros exactos), lo reformulamos como un problema **combinatorio/multi-etiqueta**:

- **Regresion** (enfoque anterior): Predecir `y = [r1, r2, ..., r6]` como valores continuos. Problema: los numeros de loteria no son valores continuos con relacion ordinal significativa.
- **Clasificacion multi-etiqueta** (enfoque mejorado): Para cada numero `k` en `[1, N]`, predecir la probabilidad `P(k aparece en el proximo sorteo)`. Luego seleccionar los `m` numeros con mayor probabilidad.

Formalmente, para cada numero `k`:

```
P(k in S_{t+1} | features_t) = f_modelo(features_t)
```

Donde `S_{t+1}` es el conjunto de numeros del sorteo `t+1`.

Este enfoque es superior porque:
1. Respeta la naturaleza combinatoria del juego (elegir `m` de `N`).
2. Produce probabilidades calibrables para cada numero.
3. Permite aplicar restricciones combinatorias naturalmente.
4. Es compatible con metricas de negocio como Hit@k.

### 5.0.1 Baseline Obligatorio: Modelo Uniforme Aleatorio

**TODO modelo debe superar este baseline** para justificar su complejidad:

```
P_baseline(k) = m / N    para todo k en [1, N]
```

Para Melate (6 de 56): `P_baseline(k) = 6/56 = 0.1071` para cada numero.

**Rendimiento esperado del baseline:**

```
E[aciertos por sorteo] = m * (m / N) = m^2 / N = 36/56 = 0.643
```

**Test de superioridad (binomial test):**

Para declarar que un modelo supera al baseline, necesitamos significancia estadistica:

```
H0: P(modelo acierta) = P(baseline acierta)
H1: P(modelo acierta) > P(baseline acierta)

p-value = P(X >= aciertos_modelo | X ~ Binomial(n_tests, p_baseline))
```

Rechazamos H0 solo si `p-value < 0.05`. Este test se aplica en **backtesting temporal** con datos que el modelo nunca vio durante entrenamiento.

### 5.1 Modelo 1: Frecuencias Bayesianas (Modelo Simple Prioritario)

#### 5.1.1 Fundamento Matematico

Antes de usar modelos complejos, comenzamos con un enfoque bayesiano simple que estima la probabilidad de cada numero usando un prior conjugado:

**Prior: Distribucion Beta**

```
P(theta_k) = Beta(alpha, beta)
```

Donde `alpha = beta = 1` (prior uniforme/no informativo).

**Posterior tras observar datos:**

```
P(theta_k | datos) = Beta(alpha + exitos_k, beta + fracasos_k)
```

Donde:
- `exitos_k` = numero de sorteos donde `k` aparecio
- `fracasos_k` = numero de sorteos donde `k` NO aparecio

**Estimacion puntual (media posterior):**

```
                     alpha + exitos_k
P_bayes(k) = ----------------------------------
              alpha + beta + exitos_k + fracasos_k
```

Con prior uniforme `(alpha=1, beta=1)`:

```
                  1 + f(k)
P_bayes(k) = ---------------
                2 + n
```

Donde `f(k)` es la frecuencia absoluta del numero `k` y `n` es el total de sorteos.

**Intervalo de credibilidad bayesiano (95%):**

```
IC_95%(theta_k) = [Beta_inv(0.025, a_post, b_post), Beta_inv(0.975, a_post, b_post)]
```

#### 5.1.2 Variante con Ventana Temporal

Para capturar tendencias recientes, usamos solo los ultimos `w` sorteos:

```
                         1 + f(k, ultimos w)
P_bayes_temporal(k) = ---------------------------
                           2 + w
```

#### 5.1.3 Por que Frecuencias Bayesianas como Primer Modelo

- **Simplicidad**: Facil de implementar, interpretar y depurar.
- **Regularizacion natural**: El prior evita probabilidades de 0 o 1 con pocos datos.
- **Baseline robusto**: Si este modelo simple no supera al uniforme, es improbable que modelos mas complejos lo hagan de forma genuina.
- **Cuantificacion de incertidumbre**: La distribucion posterior nos da intervalos de credibilidad directamente.

### 5.2 Modelo 2: Random Forest Classifier

#### 5.2.1 Fundamento Matematico

Random Forest construye un ensamble de `B` arboles de decision, cada uno entrenado con un subconjunto aleatorio de datos (bootstrap) y features.

**Reformulado como clasificacion:** En lugar de regresion, usamos `RandomForestClassifier` donde para cada numero `k`, entrenamos un clasificador binario:

```
y_k(t) = 1 si k aparecio en el sorteo t, 0 en caso contrario
```

La prediccion del ensamble es:

```
                  1   B
P(y_k=1|x) = --- SUM T_b(x)_k
                  B  b=1
```

Donde `T_b(x)_k` es la prediccion de probabilidad del arbol `b` para la clase `y_k=1`.

**Criterio de division (Gini Impurity para clasificacion):**

En cada nodo, el arbol busca la division que minimiza la impureza:

```
Gini(S) = 1 - SUM p_c^2
```

Donde `p_c` es la proporcion de la clase `c` en el nodo `S`.

**Importancia de features (Mean Decrease Impurity):**

```
                     B
Imp(feature_j) = (1/B) SUM  SUM  p(t) * delta_Gini(t, j)
                     b=1  t in T_b
```

Esto nos dice cuales features contribuyen mas a la prediccion.

#### 5.2.2 Por que Random Forest

- **Robusto a overfitting**: El promedio de multiples arboles reduce la varianza sin aumentar el sesgo.
- **No requiere normalizacion**: Funciona con features en diferentes escalas.
- **Importancia de features**: Proporciona un ranking natural de que variables son mas predictivas.
- **No linealidad**: Captura relaciones no lineales entre features y numeros.
- **Probabilidades**: `predict_proba()` devuelve probabilidades calibrables por numero.

### 5.3 Modelo 3: Gradient Boosting Classifier (XGBoost / LightGBM)

#### 5.3.1 Fundamento Matematico

Gradient Boosting construye arboles secuencialmente, donde cada arbol corrige los errores del anterior:

```
F_m(x) = F_{m-1}(x) + eta * h_m(x)
```

Donde:
- `F_m(x)` = modelo en la iteracion `m`
- `eta` = learning rate (tasa de aprendizaje)
- `h_m(x)` = arbol que predice el gradiente negativo de la funcion de perdida

**Funcion de perdida (MSE):**

```
L(y, F(x)) = (1/2) * (y - F(x))^2
```

**Gradiente negativo (pseudo-residuos):**

```
r_im = -dL/dF(x_i) = y_i - F_{m-1}(x_i)
```

Cada nuevo arbol `h_m` se entrena para predecir estos residuos.

**Regularizacion (XGBoost):**

```
Obj = SUM L(y_i, y_hat_i) + SUM [gamma * T_j + (lambda/2) * ||w_j||^2]
```

Donde:
- `gamma` = penalizacion por complejidad (numero de hojas)
- `lambda` = regularizacion L2 sobre los pesos de las hojas
- `T_j` = numero de hojas del arbol `j`
- `w_j` = pesos de las hojas

**Reformulado como clasificacion:** Usamos `XGBClassifier` con funcion de perdida de log-loss:

```
L(y, p) = -[y * log(p) + (1-y) * log(1-p)]
```

Donde `y` es la etiqueta binaria (numero aparecio o no) y `p` es la probabilidad predicha.

#### 5.3.2 Por que Gradient Boosting

- **Mayor precision**: Generalmente supera a Random Forest en datasets tabulares.
- **Regularizacion integrada**: Controla overfitting con `gamma`, `lambda` y `eta`.
- **Manejo de missing values**: XGBoost aprende automaticamente la direccion optima para nulos.
- **Probabilidades calibradas**: La funcion sigmoide de salida produce probabilidades directamente.

### 5.4 ~~Modelo 4: Red Neuronal LSTM~~ [ELIMINADO en v2]

> **Eliminado (v2):** LSTM requiere tensorflow/keras, tiene ordenes de magnitud mas parametros que los demas modelos, necesita mucho mas tuning, y para ~5,000 filas no hay suficientes datos para entrenar una red recurrente de forma robusta. Si existe autocorrelacion significativa (Paso 3.5.3), Gradient Boosting con features temporales (tendencia, rezago, ciclo_medio, aceleracion) ya captura la misma informacion temporal de forma mas eficiente. Se elimina para simplificar el stack y reducir la complejidad innecesaria.

### 5.5 Modelo 5: Cadenas de Markov

#### 5.5.1 Fundamento Matematico

Modelamos la transicion entre estados (numeros) como un proceso de Markov:

```
P(X_{n+1} = j | X_n = i) = p_ij
```

La **matriz de transicion** `P` tiene dimensiones `N x N`:

```
P = | p_11  p_12  ...  p_1N |
    | p_21  p_22  ...  p_2N |
    | ...   ...   ...  ...  |
    | p_N1  p_N2  ...  p_NN |
```

Donde `p_ij` se estima como:

```
           count(i seguido por j)
p_ij = ----------------------------
           count(i aparece)
```

**Distribucion estacionaria:**

Buscamos el vector `pi` tal que:

```
pi * P = pi       y       SUM pi_i = 1
```

Este vector nos da la probabilidad a largo plazo de cada numero, independiente del estado inicial.

**Elevacion de la matriz a la potencia n:**

```
P^n = P * P * ... * P   (n veces)
```

`P^n[i][j]` da la probabilidad de estar en el estado `j` despues de `n` pasos, comenzando en `i`. Usando autovalores:

```
P^n = Q * Lambda^n * Q^(-1)
```

Donde `Lambda` es la matriz diagonal de autovalores de `P` y `Q` la matriz de autovectores.

#### 5.5.2 Validacion Obligatoria de Dependencia Secuencial

> **Advertencia critica:** Las Cadenas de Markov asumen que el estado siguiente **depende** del estado actual (`P(X_{n+1}|X_n) != P(X_{n+1})`). En loteria, si los sorteos son independientes, la matriz de transicion deberia converger a probabilidades uniformes y el modelo no aportaria nada sobre el baseline.

**Antes de implementar este modelo, se debe validar la dependencia secuencial con:**

**Test de Rachas (Runs Test):**

Verifica si la secuencia de apariciones de cada numero es aleatoria:

```
H0: La secuencia de apariciones es aleatoria (independencia)
H1: Existe dependencia secuencial

Z_rachas = (R - E[R]) / sqrt(Var(R))
```

Donde `R` es el numero de rachas observadas.

**Test de Ljung-Box:**

Verifica si existe autocorrelacion significativa en la serie:

```
              L
Q(L) = n(n+2) SUM  rho_k^2 / (n-k)
              k=1
```

Si `Q(L) > X^2_critico(alpha, L)`, existe dependencia secuencial.

**Regla de decision:**

```
Si p_value(Runs Test) < 0.05 O p_value(Ljung-Box) < 0.05:
    -> Implementar Cadenas de Markov
Si no:
    -> Descartar Cadenas de Markov (sorteos son independientes)
```

#### 5.5.3 Por que Cadenas de Markov (si se valida dependencia)

- **Interpretabilidad**: Las probabilidades de transicion son directamente comprensibles.
- **Prediccion directa**: `P^n` nos da probabilidades de prediccion a `n` sorteos futuros.
- **Deteccion de patrones secuenciales**: Captura que numeros tienden a "seguir" a otros.

### 5.6 Reproducibilidad y Versionado

> **Correccion v2:** Se agrega como requisito transversal para todos los modelos.

#### 5.6.0.1 Semillas Aleatorias

Fijar `random_state` en **todos** los componentes estocasticos:

```python
RANDOM_STATE = 42

RandomForestClassifier(random_state=RANDOM_STATE)
GradientBoostingClassifier(random_state=RANDOM_STATE)
train_test_split(random_state=RANDOM_STATE)
np.random.seed(RANDOM_STATE)
```

#### 5.6.0.2 Versionado de Datos

Cada ejecucion registrara un hash del CSV para auditar reproducibilidad:

```
data_hash = hashlib.sha256(csv_bytes).hexdigest()[:16]
data_version = f"v{n_sorteos}_{fecha_ultimo_sorteo}_{data_hash}"
```

#### 5.6.0.3 Estrategia de Hyperparameter Tuning

En lugar de GridSearchCV estandar (que viola el orden temporal con K-Fold), usar **busqueda con TimeSeriesSplit**:

```
Para cada combinacion de hiperparametros:
    1. Crear TimeSeriesSplit(n_splits=5) sobre los datos de entrenamiento
    2. Evaluar con Brier Score en cada split temporal
    3. Promediar scores

Seleccionar hiperparametros con mejor Brier Score promedio
```

**Alternativa preferida: Optuna** (busqueda bayesiana):

```
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
    }
    score = walk_forward_evaluate(model_class(**params), data)
    return score  # Brier Score (minimizar)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 5.7 Evaluacion y Seleccion de Modelos

#### 5.6.1 Validacion Walk-Forward Estricta

En lugar de K-Fold estandar (que mezcla datos temporales), usamos **Walk-Forward Validation** que simula exactamente como se usaria el modelo en produccion:

```
Para t = T_inicio hasta T_final:
    1. Entrenar modelo con sorteos 1..t-1
    2. Predecir sorteo t
    3. Registrar aciertos
    4. Avanzar: t = t + 1
```

**Variante con ventana deslizante** (para capturar cambios de regimen):

```
Para t = T_inicio hasta T_final:
    1. Entrenar modelo con sorteos max(1, t-W)..t-1   (ventana de W sorteos)
    2. Predecir sorteo t
    3. Registrar aciertos
```

Esto es mas riguroso que Time Series Split porque:
- Evalua en **cada** sorteo, no solo en bloques.
- Nunca usa datos futuros para entrenar.
- Simula el uso real del sistema.

#### 5.6.2 Metricas de Evaluacion (Priorizadas por Relevancia)

**Metricas PRIMARIAS (metricas de negocio):**

Estas son las metricas que determinan si el modelo es util:

**Hit Rate (Tasa de Acierto):**

Metrica especifica para loterias:

```
                  sorteos donde al menos k numeros predichos fueron correctos
Hit_Rate(k) = -------------------------------------------------------------------
                                    total de sorteos evaluados
```

**Distribucion de Aciertos:**

```
Dist(j) = P(exactamente j aciertos en un sorteo)

        = count(sorteos con j aciertos) / total_sorteos
```

Compararemos esta distribucion contra la distribucion hipergeometrica esperada por azar:

```
                    C(m, j) * C(N-m, m-j)
P_azar(j aciertos) = -------------------------
                          C(N, m)
```

**ROI Simulado:**

Si el sistema se usara para jugar, el ROI estimado seria:

```
              ganancia_total - costo_total
ROI = 100 * --------------------------------
                   costo_total
```

Donde `ganancia_total` considera los premios por nivel de aciertos y `costo_total` es el numero de sorteos jugados por el costo del boleto.

**Metricas SECUNDARIAS (diagnostico del modelo):**

**Brier Score (calibracion de probabilidades):**

```
             1   n
BS = --- SUM (p_i - o_i)^2
             n  i=1
```

Donde `p_i` es la probabilidad predicha y `o_i` es el resultado observado (0 o 1). Un Brier Score bajo indica buena calibracion. Para el baseline uniforme:

```
BS_baseline = (m/N) * (1 - m/N) = 0.1071 * 0.8929 = 0.0956
```

**Log-Loss (entropia cruzada binaria):**

```
               1   n
LogLoss = - --- SUM [o_i * log(p_i) + (1-o_i) * log(1-p_i)]
               n  i=1
```

**Metricas TERCIARIAS (solo como referencia, NO para decision):**

MAE y RMSE se reportaran pero no se usaran para seleccionar modelos, ya que no son adecuadas para problemas de clasificacion multi-etiqueta:

```
MAE = (1/n) SUM |y_i - y_hat_i|
RMSE = sqrt((1/n) SUM (y_i - y_hat_i)^2)
```

#### 5.6.3 Calibracion de Probabilidades

Las probabilidades crudas de los modelos suelen estar mal calibradas. Aplicaremos:

**Platt Scaling (Regresion Logistica):**

Ajustar una sigmoide sobre las probabilidades crudas:

```
P_calibrada = 1 / (1 + exp(A * P_cruda + B))
```

Donde `A` y `B` se ajustan por maxima verosimilitud en un conjunto de validacion.

**Regresion Isotonica:**

Ajuste no parametrico que preserva el orden:

```
P_calibrada = f_isotonica(P_cruda)
```

Donde `f_isotonica` es una funcion monotona no decreciente ajustada por minimos cuadrados.

**Curva de Calibracion (Reliability Diagram):**

Divide las predicciones en bins por probabilidad predicha y compara con la frecuencia real:

```
Para cada bin b con probabilidad media p_b:
    frecuencia_real_b = count(positivos en bin b) / count(muestras en bin b)
    
    Modelo calibrado: frecuencia_real_b ~ p_b
```

#### 5.6.4 Comparacion Obligatoria contra Baseline

**Regla de decision:**

```
Si p_value(test binomial, modelo vs baseline) < 0.05:
    -> Modelo aprobado: usar para prediccion
Si no:
    -> Modelo descartado: no es mejor que el azar
    -> Adoptar enfoque descriptivo (analisis de patrones sin prediccion)
```

Adicional, compararemos contra **S=10,000 simulaciones aleatorias equivalentes**:

```
Para s = 1 hasta S:
    1. Generar predicciones aleatorias (m numeros de N)
    2. Calcular Hit_Rate de las predicciones aleatorias

percentil_modelo = P(Hit_Rate_aleatorio < Hit_Rate_modelo)

Si percentil_modelo > 0.95:
    -> Modelo supera al azar
Si no:
    -> Modelo NO supera al azar
```

### 5.7 Libreria Principal: scikit-learn

**Por que scikit-learn es la opcion optima para modelado:**

- **API unificada**: Todos los modelos siguen el patron `fit()` / `predict()` / `score()`, facilitando la comparacion.
- **Clasificadores completos**: RandomForestClassifier, GradientBoostingClassifier, cross_val_score, GridSearchCV.
- **Calibracion integrada**: `CalibratedClassifierCV` implementa Platt Scaling e Isotonic Regression.
- **Pipelines**: Encadenan preprocesamiento + modelo en un solo objeto reproducible.
- **Comunidad y documentacion**: La libreria de ML mas usada en Python con documentacion exhaustiva.
- **Alternativas complementarias**:
  - `xgboost`: Se usara como complemento para Gradient Boosting optimizado (mas rapido que sklearn GBC, soporte GPU nativo).
  - `optuna`: Hyperparameter tuning bayesiano que respeta el orden temporal.
  - ~~`tensorflow/keras`~~: Eliminado en v2 junto con LSTM.

---

## Paso 6: Sistema de Generacion de Predicciones

### 6.1 Metodo de Ensamble Final

Combinamos las predicciones de todos los modelos usando un promedio ponderado:

```
                   M
P_final(k) = SUM w_m * P_m(k)
                  m=1
```

Donde:
- `P_m(k)` = probabilidad/score del modelo `m` para el numero `k`
- `w_m` = peso del modelo `m`, determinado por su rendimiento en validacion

Los pesos se calculan usando metricas coherentes con el problema de clasificacion:

**Metodo primario — basado en Hit Rate (metrica de negocio):**

```
             Hit_Rate_m
w_m = -------------------------
       SUM Hit_Rate_j  para todo j
```

**Metodo alternativo — basado en Brier Score inverso (calibracion):**

```
             1 / BS_m
w_m = -------------------------
       SUM (1 / BS_j)  para todo j
```

Donde `BS_m` es el Brier Score del modelo `m` en walk-forward validation. Se usa Brier Score (no RMSE) porque es la metrica apropiada para evaluar probabilidades calibradas en clasificacion binaria.

> **Nota (correccion v2):** La version anterior usaba RMSE para ponderar el ensamble, lo cual era inconsistente con la definicion de RMSE como metrica terciaria "NO para decision" en el Paso 5.6.2. Se corrigio para usar metricas de clasificacion.

### 6.2 Seleccion de Combinacion Optima

Una vez que tenemos `P_final(k)` para cada numero, seleccionamos la combinacion:

1. **Top-K**: Seleccionar los `m` numeros con mayor `P_final(k)`.
2. **Muestreo ponderado**: Muestrear `m` numeros sin reemplazo usando `P_final(k)` como distribucion de probabilidad.
3. **Restricciones**: Filtrar combinaciones que no cumplan patrones historicos:
   - Rango minimo entre numeros
   - Balance par/impar dentro de limites historicos
   - Distribucion de decenas razonable

### 6.3 Simulacion de Monte Carlo

Para evaluar la robustez de las predicciones, realizamos `S` simulaciones:

```
Para s = 1 hasta S:
    1. Muestrear combinacion C_s usando P_final como distribucion
    2. Verificar C_s contra restricciones
    3. Almacenar C_s si es valida

Score_numero(k) = count(k in C_s para todo s valido) / S_valido
```

Numeros que aparecen consistentemente en las combinaciones simuladas tienen mayor confianza.

**Intervalos de confianza via bootstrap:**

```
Para b = 1 hasta B:
    1. Remuestrear con reemplazo los datos de entrenamiento
    2. Re-entrenar modelos
    3. Generar prediccion

IC(95%) = [percentil_2.5(predicciones), percentil_97.5(predicciones)]
```

### 6.4 Comparacion Final contra Simulaciones Aleatorias

Antes de aceptar cualquier prediccion como valida, la comparamos contra `S=10,000` simulaciones de seleccion puramente aleatoria:

```
Para s = 1 hasta 10,000:
    1. Seleccionar m numeros uniformemente al azar de [1, N]
    2. Calcular aciertos contra el sorteo real

Distribucion_azar = histograma(aciertos de las 10,000 simulaciones)
Percentil_modelo = P(aciertos_azar < aciertos_modelo)
```

Esto proporciona una medida empirica de si las predicciones del modelo son genuinamente superiores al azar o si los resultados observados podrian deberse a varianza aleatoria.

**Regla de decision final:**

```
Si Percentil_modelo > 95% de forma consistente en walk-forward:
    -> Sistema en modo PREDICTIVO: generar y reportar predicciones
Si no:
    -> Sistema en modo DESCRIPTIVO: reportar analisis de patrones,
       frecuencias y tendencias SIN emitir predicciones numericas
```

### 6.5 Libreria: matplotlib

**Por que matplotlib es la opcion optima para visualizacion:**

- **Control total**: Permite personalizar cada aspecto del grafico (ejes, colores, anotaciones, leyendas).
- **Versatilidad**: Soporta graficos de barras, lineas, heatmaps, scatter plots, histogramas y mas.
- **Exportacion**: Genera imagenes en PNG, PDF, SVG y otros formatos de alta calidad.
- **Base de seaborn**: Toda visualizacion estadistica de seaborn se construye sobre matplotlib.
- **Alternativas descartadas**:
  - `plotly`: Interactivo pero mas pesado y complejo. No necesario para un script de prediccion.
  - `bokeh`: Similar a plotly, orientado a dashboards web. Excesivo para nuestro caso de uso.

---

## Paso 7: Salida de Resultados y Logging

### 7.1 Formato de Salida

- **CSV**: Predicciones con probabilidades para cada numero.
- **Graficos**: Distribuciones de frecuencia, heatmaps de correlacion, tendencias temporales.
- **Reporte**: Resumen con metricas de evaluacion, features mas importantes, y combinaciones sugeridas.

### 7.2 Logging

Cada ejecucion registrara:
- Timestamp de ejecucion
- Version de los datos (numero de sorteos procesados)
- Hiperparametros de cada modelo
- Metricas de evaluacion
- Predicciones generadas

---

## Resumen del Stack Tecnologico

| Libreria | Version | Uso Principal | Justificacion |
|----------|---------|---------------|---------------|
| **pandas** | 3.0.1 | Carga, limpieza y manipulacion de datos | API optima para datos tabulares, operaciones vectorizadas |
| **numpy** | 2.4.3 | Calculos matematicos y estadisticos | Operaciones vectorizadas en C, base del ecosistema |
| **scipy** | 1.17.1 | Tests estadisticos, distribuciones, calibracion | Tests con p-values exactos, funciones especiales, distribucion Beta |
| **scikit-learn** | 1.8.0 | Modelos ML, validacion, calibracion, preprocesamiento | API unificada, CalibratedClassifierCV, pipelines, RFE |
| **xgboost** | latest | Gradient Boosting optimizado con GPU | Mas rapido que sklearn GBC, soporte GPU nativo |
| **optuna** | latest | Hyperparameter tuning bayesiano | Busqueda eficiente respetando orden temporal |
| **matplotlib** | 3.10.8 | Graficos y visualizacion | Control total, formatos multiples, base de seaborn |
| **seaborn** | 0.13.2 | Visualizacion estadistica | Graficos estadisticos elegantes, integracion pandas |
| **requests** | 2.33.0 | Obtencion de datos remotos (si aplica) | Libreria HTTP mas usada, API simple y confiable |
| **joblib** | 1.5.3 | Persistencia de modelos entrenados | Serializacion eficiente de objetos sklearn/numpy |

---

## Orden de Ejecucion del Script

```
1. Obtener CSV  -->  2. Validar datos  -->  3. EDA  -->  4. Features (con validacion temporal)
                                                              |
                                                              v
                                                    4.5 Seleccion formal de features
                                                        (MI, RFE, Boruta-like)
                                                              |
                                                              v
                                                    4.6 Validar features
                                                        (sin fuga temporal)
                                                              |
                                                              v
                                                    5.0 Manejo desbalance clases
                                                              |
                                                              v
                                                    5.0.1 Baseline uniforme
                                                              |
                                                              v
9. Exportar  <--  8. Evaluar vs baseline  <--  7. Entrenar modelos
     |                     |                   (Bayesiano -> RF -> GB
     v                     |                    -> Markov si test OK)
  Modo PREDICTIVO          |                         |
  o DESCRIPTIVO   <--------+                   6. Tuning (Optuna)
  (segun resultado                                   |
   vs baseline)                              5.6 Reproducibilidad
                                               (random_state, hash)
```

---

## Orden de Prioridad de Modelos

Los modelos se implementaran y evaluaran en este orden estricto. Solo se avanza al siguiente si el anterior no supera al baseline:

| Prioridad | Modelo | Complejidad | Justificacion | Prerequisito |
|-----------|--------|-------------|---------------|--------------|
| 1 | Baseline Uniforme | Nula | Punto de referencia obligatorio | Ninguno |
| 2 | Frecuencias Bayesianas | Baja | Regularizacion natural, incertidumbre cuantificada | Ninguno |
| 3 | Random Forest Classifier | Media | Robusto, importancia de features, seleccion de features | `class_weight='balanced'` |
| 4 | Gradient Boosting Classifier | Media-Alta | Mayor precision en datos tabulares | `class_weight='balanced'`, tuning con Optuna |
| 5 | Cadenas de Markov | Media | Patrones secuenciales, interpretable | **Solo si Runs Test o Ljung-Box p < 0.05** |
| ~~6~~ | ~~LSTM~~ | ~~Alta~~ | ~~Eliminado en v2: insuficientes datos, complejidad injustificada~~ | N/A |

## Proximos Pasos

1. **Implementar Paso 1**: Modulo `data_service.py` para obtener y validar el CSV historico.
2. **Definir fuente del CSV**: Determinar si el archivo sera local, descargado de una URL, o via API.
3. **Comenzar EDA**: Implementar el analisis exploratorio en un Jupyter Notebook para explorar los datos antes de automatizar.
4. **Implementar Baseline**: Crear el modelo uniforme aleatorio como primera referencia de comparacion.
