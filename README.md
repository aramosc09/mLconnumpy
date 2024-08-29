# Fetal Health Classification

Este proyecto es un sistema de clasificación para la salud fetal utilizando un modelo de regresión logística. El código procesa un conjunto de datos, entrena un modelo de regresión logística y predice la salud fetal con base en nuevos datos de entrada.

## Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes librerías de Python:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

En caso de no tenerlas, me aseguré que al correr el proyecto se instalen gracias a `requirements.txt`.

## Archivos necesarios

El proyecto requiere los siguientes archivos:

- `fetal_health.csv`: Un archivo CSV que contiene los datos del conjunto de características y la columna objetivo `fetal_health`.

- `data_description.json`: Un archivo JSON que contiene descripciones de cada una de las características del conjunto de datos.

Asegúrate de que estos archivos estén en las rutas especificadas o modifica las rutas en el código.

## Uso

### 1. Obtener y preparar el dataset
La función `get_fetal_dataset(file_path, corr_threshold)` realiza las siguientes operaciones:

- Carga los datos desde el archivo CSV.

- Calcula la correlación de las características con la columna objetivo `fetal_health` y selecciona las características con una correlación superior al umbral `(corr_threshold)`.

- Convierte la columna objetivo fetal_health a una variable binaria (1 para sospechoso, 0 para normal).

- Retorna las características seleccionadas, los datos, y las descripciones de las columnas.

### 2. Generar gráficos de distribución
La función `get_distribution_plots(data, cols, data_description)` genera y guarda gráficos de distribución para las columnas seleccionadas.

### 3. Entrenamiento del modelo
La función `train_rl(X, y, learning_rate, iterations, data_description)`:

- Entrena un modelo de regresión logística utilizando descenso de gradiente.

- Divide los datos en conjuntos de entrenamiento y prueba.

- Normaliza las características.

- Genera y guarda una matriz de confusión.

- Imprime la precisión del modelo en el conjunto de prueba.

### 4. Realizar predicciones
La función `predict(X, weights, scaler)` permite al usuario ingresar nuevos datos y predecir la salud fetal utilizando el modelo entrenado.

### 5. Ejecución del programa
El script principal se encuentra en la función `main()`. Para ejecutar todo el proceso, simplemente corre el archivo en el directorio correcto:

```bash
python main.py
```

## Personalización
Puedes ajustar el umbral de correlación, la posición del sigmoide, la tasa de aprendizaje `(learning_rate)`, y el número de iteraciones `(iterations)` según tus necesidades en el archivo `PROJECT_PATH/input_data.json`.

## Salida
- Gráficos de distribución: Guardados como archivos PNG como `/PROJECT_PATH/graphs/distribution_<column_name>.png`.

- Matriz de confusión: Guardada como `/PROJECT_PATH/graphs/confusion_matrix_learning_rate=<learning_rate>_iterations=<iterations>.png`.

- Precisión del modelo: Impresa en la consola.

- Predicción de la salud fetal: Impresa en la consola tras la entrada de nuevos datos.