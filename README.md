# 🤖 Clasificador de Moda con PyTorch (Fashion-MNIST)

¡Hola! Este es un proyecto introductorio al mundo de la visión por computadora. El objetivo es entrenar una red neuronal sencilla con PyTorch para que aprenda a clasificar imágenes de ropa del famoso dataset **Fashion-MNIST**. Es el "Hola, Mundo" de la clasificación de imágenes.

El modelo es capaz de mirar una imagen en blanco y negro de 28x28 píxeles de una prenda y decir si es una camiseta, un pantalón, un abrigo, etc.


<img width="1867" height="945" alt="image" src="https://github.com/user-attachments/assets/c26d43d9-86f6-49e4-a339-fd64bf30c0bd" />


---

### 🛠️ Tecnologías Utilizadas

*   **Python 3.10+** 🐍
*   **PyTorch** 🔥 - El framework de Deep Learning para construir y entrenar la red.
*   **Torchvision** - Para cargar y transformar el dataset Fashion-MNIST fácilmente.
*   **Matplotlib** 📊 - Para visualizar las imágenes y los resultados de las predicciones.

---

### 📂 Estructura del Proyecto

```
/
├── data/                 # Se crea automáticamente para descargar el dataset.
├── main.py               # Script principal para ENTRENAR el modelo.
├── test.py         # Script para CARGAR y PROBAR el modelo ya entrenado.
├── model.pth             # El modelo entrenado (se genera al ejecutar main.py).
└── README.md
```

---

### 🚀 Ejecutar

Sigue estos pasos para poner en marcha el proyecto en tu máquina local.

#### 1. Clona el Repositorio
```bash
git clone https://URL_DE_TU_REPOSITORIO.git
cd nombre-del-directorio
```

#### 2. Crea un Entorno Virtual (Recomendado)
Es una buena práctica para mantener las dependencias del proyecto aisladas.

```bash
# Crear el entorno
python -m venv ml_fashion_venv

# Activarlo (en Linux/macOS)
source ml_fashion_venv/bin/activate

# Activarlo (en Windows)
.\ml_fashion_venv\Scripts\activate
```

#### 3. Instala las Dependencias
```bash
pip install torch torchvision matplotlib
```

#### 4. Scripts

**Paso 1: Entrenar el modelo**

Ejecuta `main.py` para comenzar el proceso de entrenamiento. El script descargará el dataset, entrenará la red neuronal durante 10 épocas y guardará los pesos aprendidos en el archivo `model.pth`.

```bash
python main.py
```
Verás en la consola cómo la pérdida (loss) disminuye y la precisión (accuracy) aumenta con cada época.

**Paso 2: Probar el modelo**

Una vez que tengas el archivo `model.pth`, puedes ejecutar `test.py` para ver tu modelo en acción. Este script cargará el modelo entrenado y lo usará para predecir un lote de imágenes del conjunto de prueba.

```bash
python test.py
```
Aparecerá una ventana de Matplotlib mostrando las imágenes, la predicción del modelo y si fue un acierto (verde) o un error (rojo).

---

### 📈 Rendimiento del Modelo

Tras 10 épocas de entrenamiento con la configuración actual, el modelo alcanza una **precisión de alrededor del 88%** en el conjunto de prueba.
