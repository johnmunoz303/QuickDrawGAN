# QuickDraw GAN

Aplicación para entrenar y analizar una red generativa adversaria (GAN) con dibujos de Quick, Draw!. El objetivo es observar cómo aprenden el generador y el discriminador, analizar sus parámetros y pérdidas, y generar dibujos de la categoría **smiley face (cara sonriente)**.

## Funcionamiento

La GAN está formada por dos redes que se entrenan de manera alternada:

- **Generador:** recibe un vector de ruido aleatorio y produce una imagen de 28 × 28 píxeles. Aprende a crear dibujos que el discriminador considere reales.
- **Discriminador:** recibe imágenes reales o generadas y aprende a distinguir su procedencia.

Quick, Draw! aporta los dibujos de entrenamiento. A diferencia del juego original, esta aplicación genera imágenes de una categoría y no clasifica dibujos entre distintas categorías.

## Instalación y ejecución

Requisitos: Windows, Python 3.11 o posterior y conexión a internet para instalar las dependencias y descargar los datos.

Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La aplicación se abre en el navegador en `http://localhost:8501`. La terminal debe permanecer abierta durante su uso. Para detenerla, se utiliza `Ctrl + C`.

## Uso de la aplicación

1. Seleccionar la dimensión del ruido, las épocas, la cantidad de dibujos, el tamaño del lote y las tasas de aprendizaje.
2. Pulsar **Descargar datos y entrenar**.
3. Consultar las curvas de pérdida y la evolución de los dibujos en **Entrenar y generar**.
4. Al finalizar, utilizar **Generar 16 dibujos nuevos** para obtener nuevas muestras.
5. Revisar los pesos y sesgos en **Inspeccionar valores** y el código de las redes en **Código comentado**.

La configuración inicial utiliza 3000 dibujos, 20 épocas, lotes de 64 y tasas de aprendizaje de 0,0002. El entrenamiento se realiza en CPU y su duración depende del equipo. Los controles laterales configuran el siguiente entrenamiento; la configuración del último modelo entrenado se muestra junto a sus resultados.

## Arquitectura y parámetros

### Generador

`Ruido z → Dense(128) → Dense(256) → Dense(784) → imagen 28 × 28`

Utiliza LeakyReLU entre las capas y Tanh en la salida. Con una dimensión de ruido de 64, tiene **242832 parámetros aprendibles**.

### Discriminador

`Imagen de 784 píxeles → Dense(256) → Dense(128) → Dense(1)`

Utiliza LeakyReLU entre las capas y devuelve un logit. La función sigmoid transforma ese valor en una puntuación entre 0 y 1. Tiene **233985 parámetros aprendibles**.

Los parámetros aprendibles son los pesos y sesgos. Los hiperparámetros son las opciones que controlan el entrenamiento: dimensión del ruido, épocas, lote y tasas de aprendizaje. Ambas redes utilizan Adam con beta1 = 0,5 y beta2 = 0,999. Los píxeles se normalizan de [0, 255] a [-1, 1].

## Funciones de pérdida

En las siguientes expresiones, D representa la puntuación sigmoid del discriminador:

- **Discriminador:** `L_D = -E[log D(x)] - E[log(1 - D(G(z)))]`.
- **Generador:** `L_G = -E[log D(G(z))]`.

La implementación usa `BCEWithLogitsLoss` para calcular estas pérdidas de forma numéricamente estable. El discriminador busca asignar 1 a los dibujos reales y 0 a los generados. El generador utiliza la pérdida no saturante y busca que sus dibujos reciban una puntuación cercana a 1.

En cada lote se actualiza primero el discriminador y después el generador. Durante la actualización del discriminador se desconecta el gradiente del generador; durante la actualización del generador se congelan los parámetros del discriminador.

## Valores y resultados

La interfaz muestra las pérdidas G y D por época, las puntuaciones D(real) y D(G(z)), y una galería de dibujos generados. También permite consultar la forma, cantidad, media, desviación, mínimo, máximo y los primeros 100 valores de cada tensor de pesos o sesgos.

Los promedios se ponderan por la cantidad de ejemplos de cada lote. Las puntuaciones D(real) y D(G(z)) se calculan antes de actualizar D en cada lote; la pérdida de G se calcula después de esa actualización.

Durante el entrenamiento se utiliza el mismo ruido para comparar la evolución de las imágenes. El botón de generación utiliza ruido nuevo. Se pueden descargar las métricas en CSV y los pesos junto con la configuración en un archivo `.pt`.

## Estructura

- `app.py`: interfaz de Streamlit, controles, visualización y exportación.
- `gan.py`: definición de las redes, descarga y preparación de datos, funciones de pérdida y entrenamiento. Los comentarios identifican dónde se crea cada red y explican las actualizaciones.
- `requirements.txt`: dependencias del proyecto.
- `data/`: datos descargados y almacenados localmente al ejecutar el entrenamiento.

## Alcance y limitaciones

El proyecto entrena una GAN de una sola categoría desde cero. No incluye pesos preentrenados. Los resultados dependen del entrenamiento y pueden presentar trazos borrosos, inestabilidad o poca diversidad. Una pérdida baja no garantiza calidad visual, y la puntuación del discriminador no es una medida calibrada de calidad.

Se utilizan las primeras N imágenes del archivo público, barajadas en cada época. No se incluye una evaluación con un conjunto de prueba ni una medición de generalización.
