# Plan Matematico y Tecnico - Prediccion MLT

## Introduccion

Este documento detalla el plan tecnico y matematico para construir un sistema de prediccion de resultados del MLT (Melate) basado en datos historicos. Se describen las formulas matematicas, procesos estadisticos, algoritmos de Machine Learning y la justificacion de cada libreria seleccionada.

> **Nota importante:** Los sorteos de loteria son eventos estocasticos independientes. Ningun modelo puede garantizar predicciones exactas. El objetivo de este proyecto es aplicar tecnicas estadisticas y de ML para identificar patrones, tendencias y distribuciones que permitan generar combinaciones con mayor fundamento analitico que la seleccion aleatoria.

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

### 4.4 Libreria: scikit-learn (sklearn)

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

### 5.1 Modelo 1: Random Forest Regressor

#### 5.1.1 Fundamento Matematico

Random Forest construye un ensamble de `B` arboles de decision, cada uno entrenado con un subconjunto aleatorio de datos (bootstrap) y features:

```
                  1   B
f_hat(x) = --- SUM T_b(x)
                  B  b=1
```

Donde `T_b(x)` es la prediccion del arbol `b`.

**Criterio de division (MSE):**

En cada nodo, el arbol busca la division que minimiza:

```
MSE = (1/n_L) * SUM (y_i - y_barra_L)^2  +  (1/n_R) * SUM (y_j - y_barra_R)^2
```

Donde `L` y `R` son los subconjuntos izquierdo y derecho.

**Importancia de features (Mean Decrease Impurity):**

```
                     B
Imp(feature_j) = (1/B) SUM  SUM  p(t) * delta_MSE(t, j)
                     b=1  t in T_b
```

Esto nos dice cuales features contribuyen mas a la prediccion.

#### 5.1.2 Por que Random Forest

- **Robusto a overfitting**: El promedio de multiples arboles reduce la varianza sin aumentar el sesgo.
- **No requiere normalizacion**: Funciona con features en diferentes escalas.
- **Importancia de features**: Proporciona un ranking natural de que variables son mas predictivas.
- **No linealidad**: Captura relaciones no lineales entre features y numeros.

### 5.2 Modelo 2: Gradient Boosting (XGBoost / LightGBM)

#### 5.2.1 Fundamento Matematico

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

#### 5.2.2 Por que Gradient Boosting

- **Mayor precision**: Generalmente supera a Random Forest en datasets tabulares.
- **Regularizacion integrada**: Controla overfitting con `gamma`, `lambda` y `eta`.
- **Manejo de missing values**: XGBoost aprende automaticamente la direccion optima para nulos.

### 5.3 Modelo 3: Red Neuronal LSTM (Long Short-Term Memory)

#### 5.3.1 Fundamento Matematico

LSTM es una arquitectura de red neuronal recurrente diseñada para aprender dependencias a largo plazo en secuencias:

**Puerta de Olvido (Forget Gate):**

```
f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
```

**Puerta de Entrada (Input Gate):**

```
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
C_tilde_t = tanh(W_C * [h_{t-1}, x_t] + b_C)
```

**Actualizacion del Estado de Celda:**

```
C_t = f_t (*) C_{t-1} + i_t (*) C_tilde_t
```

Donde `(*)` denota multiplicacion elemento a elemento (Hadamard product).

**Puerta de Salida (Output Gate):**

```
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t (*) tanh(C_t)
```

Donde:
- `sigma` = funcion sigmoide: `sigma(z) = 1 / (1 + e^(-z))`
- `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
- `W_f, W_i, W_C, W_o` = matrices de pesos
- `b_f, b_i, b_C, b_o` = vectores de sesgo

#### 5.3.2 Por que LSTM

- **Memoria a largo plazo**: Puede capturar patrones ciclicos que se extienden por cientos de sorteos.
- **Procesamiento secuencial**: Trata los sorteos como una serie temporal, manteniendo contexto.
- **Flexibilidad**: Puede modelar relaciones temporales complejas y no lineales.

**Nota:** Evaluaremos si la complejidad de LSTM se justifica frente a los modelos de ensamble. Si los datos no muestran patrones temporales fuertes, Random Forest o Gradient Boosting seran preferibles por su simplicidad y eficiencia.

### 5.4 Modelo 4: Cadenas de Markov

#### 5.4.1 Fundamento Matematico

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

#### 5.4.2 Por que Cadenas de Markov

- **Interpretabilidad**: Las probabilidades de transicion son directamente comprensibles.
- **Prediccion directa**: `P^n` nos da probabilidades de prediccion a `n` sorteos futuros.
- **Deteccion de patrones secuenciales**: Captura que numeros tienden a "seguir" a otros.

### 5.5 Evaluacion y Seleccion de Modelos

#### 5.5.1 Validacion Cruzada (K-Fold)

```
               1   K
CV_score = --- SUM Score(M, D_train_k, D_test_k)
               K  k=1
```

Usaremos **Time Series Split** (variante de K-Fold que respeta el orden temporal):
- Fold 1: train = sorteos 1..100, test = sorteos 101..150
- Fold 2: train = sorteos 1..150, test = sorteos 151..200
- etc.

#### 5.5.2 Metricas de Evaluacion

**Error Absoluto Medio (MAE):**

```
           1   n
MAE = --- SUM |y_i - y_hat_i|
           n  i=1
```

**Raiz del Error Cuadratico Medio (RMSE):**

```
              ___________________________
             / 1   n
RMSE = sqrt| --- SUM (y_i - y_hat_i)^2 |
             \ n  i=1                    /
```

**Coeficiente de Determinacion (R^2):**

```
              SUM (y_i - y_hat_i)^2
R^2 = 1 - -------------------------
              SUM (y_i - y_barra)^2
```

- `R^2 = 1`: prediccion perfecta
- `R^2 = 0`: el modelo no es mejor que predecir la media
- `R^2 < 0`: el modelo es peor que predecir la media

**Hit Rate (Tasa de Acierto):**

Metrica especifica para loterias:

```
                  sorteos donde al menos k numeros predichos fueron correctos
Hit_Rate(k) = -------------------------------------------------------------------
                                    total de sorteos evaluados
```

### 5.6 Libreria Principal: scikit-learn

**Por que scikit-learn es la opcion optima para modelado:**

- **API unificada**: Todos los modelos siguen el patron `fit()` / `predict()` / `score()`, facilitando la comparacion.
- **Modelos completos**: RandomForestRegressor, GradientBoostingRegressor, cross_val_score, GridSearchCV.
- **Pipelines**: Encadenan preprocesamiento + modelo en un solo objeto reproducible.
- **Comunidad y documentacion**: La libreria de ML mas usada en Python con documentacion exhaustiva.
- **Alternativas complementarias**:
  - `xgboost`: Se usara como complemento para Gradient Boosting optimizado (mas rapido que sklearn GBR).
  - `tensorflow/keras`: Solo si LSTM demuestra ser superior en la evaluacion; no se incluira por defecto para mantener el stack simple.

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

Los pesos se calculan inversamente proporcionales al error:

```
             1 / RMSE_m
w_m = -------------------------
       SUM (1 / RMSE_j)  para todo j
```

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

### 6.4 Libreria: matplotlib

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
| **scipy** | 1.17.1 | Tests estadisticos, distribuciones | Tests con p-values exactos, funciones especiales |
| **scikit-learn** | 1.8.0 | Modelos ML, validacion, preprocesamiento | API unificada, modelos robustos, pipelines |
| **matplotlib** | 3.10.8 | Graficos y visualizacion | Control total, formatos multiples, base de seaborn |
| **seaborn** | 0.13.2 | Visualizacion estadistica | Graficos estadisticos elegantes, integracion pandas |
| **requests** | 2.33.0 | Obtencion de datos remotos (si aplica) | Libreria HTTP mas usada, API simple y confiable |
| **joblib** | 1.5.3 | Persistencia de modelos entrenados | Serializacion eficiente de objetos sklearn/numpy |

---

## Orden de Ejecucion del Script

```
1. Obtener CSV  -->  2. Validar datos  -->  3. EDA  -->  4. Features
                                                              |
                                                              v
7. Exportar  <--  6. Generar prediccion  <--  5. Entrenar modelos
```

---

## Proximos Pasos

1. **Implementar Paso 1**: Modulo `data_service.py` para obtener y validar el CSV historico.
2. **Definir fuente del CSV**: Determinar si el archivo sera local, descargado de una URL, o via API.
3. **Comenzar EDA**: Implementar el analisis exploratorio en un Jupyter Notebook para explorar los datos antes de automatizar.
