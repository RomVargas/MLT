# Resumen y recomendaciones del PLAN_MATEMATICO_MLT
1) El plan es solido en ingenieria de datos: validacion del CSV, EDA y pipeline reproducible.
2) Es completo en estadistica (frecuencias, chi-cuadrado, tendencias, autocorrelacion) y cubre varias familias de modelos.
3) Punto critico: en loteria los sorteos son casi independientes; no hay garantia de mejora material frente al azar.
4) Por eso, no lo considero “el mas optimo” aun: mezcla complejidad alta (LSTM/Markov) con ganancia incierta.
5) Cambio propuesto: usar primero un baseline uniforme y comparar TODO contra ese baseline en backtesting temporal.
6) Cambio propuesto: modelar como problema combinatorio/multietiqueta (6 de N), no como regresion de numeros.
7) Cambio propuesto: optimizar una metrica de negocio real (Hit@k, distribucion de aciertos, ROI simulado), no solo MAE/RMSE.
8) Cambio propuesto: calibrar probabilidades (Platt/Isotonic) y verificar confiabilidad (Brier score, calibration curve).
9) Cambio propuesto: reducir riesgo de sobreajuste con walk-forward estricto y ventanas deslizantes.
10) Cambio propuesto: eliminar features con fuga temporal y validar su estabilidad por periodos.
11) Cambio propuesto: priorizar modelos simples/robustos (frecuencias bayesianas + Gradient Boosting) antes de LSTM.
12) Cambio propuesto: reportar incertidumbre por bootstrap y comparar contra simulaciones aleatorias equivalentes.
13) Si no supera al baseline de forma estadisticamente significativa, mantener enfoque descriptivo y no predictivo.
14) Conclusion: muy buen plan tecnico; para precision real debe simplificarse, calibrarse y evaluarse con rigor causal-temporal.
