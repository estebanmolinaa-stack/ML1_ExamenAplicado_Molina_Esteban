# Predicción del volumen de tráfico horario en la autopista I-94

**Examen Aplicado de Machine Learning I**
Esteban Molina Almazabal
Ingeniería en Inteligencia Artificial, Universidad Mayor

---

## 1. Descripción del dataset

| Campo | Valor |
|---|---|
| Nombre | Metro Interstate Traffic Volume |
| Fuente | UCI Machine Learning Repository, dataset 492 |
| URL | https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume |
| Descarga directa | https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip |
| Observaciones | 48.204 originales, 40.575 tras deduplicar |
| Variables | 9 originales, 11 predictoras tras la ingeniería de variables |
| Variable objetivo | `traffic_volume`, vehículos por hora |
| **Tipo de tarea** | **Regresión** |
| Licencia | Creative Commons Attribution 4.0 |

Los datos provienen de la estación de conteo ATR 301 de la autopista Interestatal 94 en
dirección oeste, entre Minneapolis y St. Paul, Estados Unidos. Cada fila corresponde a
una hora e informa cuántos vehículos circularon, junto con las condiciones
meteorológicas de esa hora y si el día era feriado. El período va de octubre de 2012 a
septiembre de 2018.

## 2. Metodología

**Análisis exploratorio y limpieza.** Se detectaron tres problemas de calidad de datos
que condicionaron todo el trabajo:

* Los 48.143 valores faltantes que pandas reporta en `holiday` son un artefacto de
  lectura. La columna usa el texto literal `"None"` como categoría y el dataset no tiene
  ningún nulo real.
* 7.629 marcas temporales duplicadas, porque la fuente escribe una fila por condición
  meteorológica simultánea. Se conserva el primer registro de cada hora.
* Once registros con valores físicamente imposibles: 10 con `temp` igual a 0 Kelvin y 1
  con `rain_1h` de 9.831 mm en una hora. Se marcan como `NaN` para que los impute el
  pipeline con la mediana de entrenamiento.

**Ingeniería de variables.** Las variables meteorológicas casi no correlacionan con el
tráfico, mientras que el volumen pasa de 373 vehículos a las 3 de la madrugada a 5.709 a
las 16 horas. Se extraen `hour`, `weekday`, `month`, `is_weekend` e `is_holiday` desde
`date_time`.

**Preprocesamiento sin fuga de información.** La partición train/test 80/20 se hace
antes de ajustar cualquier transformador. El `ColumnTransformer` tiene tres bloques:
mediana más `StandardScaler` para las continuas, moda más `OrdinalEncoder` más
`StandardScaler` para las ordinales, y moda más `OneHotEncoder` para las nominales. Se
ajusta con `.fit_transform(X_train)` y al test solo se le aplica `.transform()`.

**Análisis no supervisado.** PCA necesita 7 componentes para el 85,65% de la varianza.
K-Means entregó al inicio un cluster degenerado de 9 observaciones, causado por
`snow_1h`, que vale cero en el 99,92% de los registros y al estandarizarse produce
z-scores de hasta 83. Excluida esa variable del espacio de segmentación, K = 4 entrega
grupos equilibrados que se organizan por estación del año, nubosidad y tipo de día.

**Modelado.** Tres modelos con `random_state=42`: un baseline que predice la media,
Ridge con `alpha` optimizado por `GridSearchCV` de 5 folds, y Random Forest con
`n_estimators`, `max_depth` y `min_samples_split` optimizados por `GridSearchCV` de 5
folds.

## 3. Resultados

| Modelo | RMSE test | MAE test | R2 test | MAPE test | R2 train | Brecha R2 | Entrenamiento | Inferencia |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **385,7384** | **237,5301** | **0,9620** | 45,8347 | 0,9825 | 0,0205 | 21,79 s | 0,2658 s |
| Ridge (L2) | 1766,1594 | 1545,6485 | 0,2037 | 164,7442 | 0,2045 | 0,0008 | 0,0098 s | 0,0013 s |
| Baseline (media) | 1979,3329 | 1738,0349 | -0,0001 | 218,0661 | 0,0000 | 0,0001 | 0,0 s | 0,0 s |

**Modelo seleccionado: Random Forest** con 300 árboles, `max_depth=20` y
`min_samples_split=10`.

El RMSE de 385,74 vehículos por hora es un 78,2% menor que el de Ridge y un 80,5% menor
que el del baseline. La brecha de R2 entre train y test es de 0,0205, de modo que no hay
sobreajuste relevante.

**Variables más importantes:** `hour` con 85,06%, `is_weekend` con 5,93%, `weekday` con
4,54%, `temp` con 1,42% y `month` con 0,89%. Las cinco concentran el 96,95% de la
capacidad predictiva.

Nota sobre el MAPE: los valores son altos y no deben leerse como error porcentual real,
porque la variable objetivo alcanza mínimos de 2 vehículos por hora y esos casos dominan
el promedio. La métrica principal del trabajo es el RMSE.

## 4. Estructura del repositorio

```
ML1_ExamenAplicado_Molina_Esteban/
├── ML1_ExamenAplicado_Molina_Esteban.ipynb   Notebook ejecutado
├── README.md
├── requirements.txt
├── resultados_modelos.csv                    Tabla comparativa exportada
├── data/raw/                                 Dataset original sin modificar
└── figures/                                  17 figuras a 150 dpi
```

## 5. Video de presentación

Enlace: PENDIENTE

## 6. Reproducir el análisis

```bash
pip install -r requirements.txt
jupyter notebook ML1_ExamenAplicado_Molina_Esteban.ipynb
```

El notebook incluye un interruptor `EJECUCION_LOCAL` para alternar entre ejecución local
y Google Colab. El dataset ya está incluido en `data/raw/`, de modo que no requiere
descarga adicional. Todos los procesos usan `SEED = 42`, por lo que los resultados son
reproducibles.

## 7. Declaración de uso de inteligencia artificial generativa

Conforme a lo exigido en la ficha de examen, se declara el uso de IA generativa en este
trabajo.

**Herramienta utilizada:** Claude (Anthropic), a través de Claude Code.

**En qué se utilizó:**

* Apoyo en la exploración inicial del dataset y en la verificación de que cumpliera los
  requisitos de la ficha.
* Redacción y estructuración del código del notebook, incluyendo el pipeline de
  preprocesamiento, la búsqueda de hiperparámetros y las visualizaciones.
* Apoyo en la redacción de las secciones de análisis e interpretación en Markdown.
* Diagnóstico del cluster degenerado de K-Means y de su causa en la varianza de
  `snow_1h`.

**En qué no se utilizó:**

* La selección del dataset y la definición del enfoque del trabajo fueron decisiones
  propias.
* Todos los resultados numéricos reportados provienen de la ejecución real del código
  sobre los datos, y no fueron generados ni estimados por la herramienta.

**Verificación realizada:** el notebook fue ejecutado de principio a fin y todas las
cifras citadas en el texto corresponden a las salidas efectivas de las celdas.
