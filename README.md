# Clasificador de Figuras Geométricas con Red Neuronal

Este proyecto implementa un sistema completo de clasificación de figuras geométricas (círculo, cuadrado y triángulo) utilizando una red neuronal artificial (Perceptrón Multicapa).

## 🎯 Características del Proyecto

- **Generación automática de dataset** sintético con variaciones realistas
- **Extracción de características geométricas** (no clasificación directa de imágenes)
- **Red neuronal MLP** entrenada con TensorFlow/Keras
- **Interfaz web** interactiva con Flask y Bootstrap
- **Métricas completas** de evaluación y visualizaciones

## 🔧 Requisitos

- Python 3.8 o superior
- Ver `requirements.txt` para todas las dependencias

## 📦 Instalación

1. Instala las dependencias:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 🚀 Uso del Sistema

### Paso 1: Generar el Dataset

Crea imágenes sintéticas de las tres figuras geométricas:

\`\`\`bash
python scripts/generar_dataset.py
\`\`\`

Esto generará 500 imágenes de cada clase en la carpeta `dataset/`:
- `dataset/circulo/` - 500 imágenes de círculos
- `dataset/cuadrado/` - 500 imágenes de cuadrados
- `dataset/triangulo/` - 500 imágenes de triángulos

### Paso 2: Extraer Características

Procesa las imágenes y extrae las características geométricas:

\`\`\`bash
python scripts/extraer_caracteristicas.py
\`\`\`

Esto crea el archivo `caracteristicas_dataset.csv` con las siguientes columnas:
- `numero_lados`: Aproximación de lados del contorno
- `proporcion_ancho_alto`: Relación ancho/alto del rectángulo delimitador
- `perimetro`: Perímetro del contorno
- `area`: Área del contorno
- `compacidad`: 4π × área / perímetro²
- `clase`: Etiqueta de la figura (circulo, cuadrado, triangulo)

### Paso 3: Entrenar el Modelo

Entrena la red neuronal con las características extraídas:

\`\`\`bash
python scripts/entrenar_modelo.py
\`\`\`

Este script:
- Divide el dataset en entrenamiento (80%) y prueba (20%)
- Entrena un Perceptrón Multicapa con 3 capas ocultas
- Genera métricas: accuracy, matriz de confusión, gráficas
- Guarda el modelo entrenado en `modelo/`

**Archivos generados:**
- `modelo/clasificador_figuras.keras` - Modelo entrenado
- `modelo/scaler.pkl` - Normalizador de características
- `modelo/label_encoder.pkl` - Codificador de etiquetas
- `matriz_confusion.png` - Visualización de la matriz de confusión
- `historial_entrenamiento.png` - Gráficas de loss y accuracy

### Paso 4: Ejecutar la Aplicación Web

Inicia el servidor Flask:

\`\`\`bash
python app.py
\`\`\`

Abre tu navegador en: `http://localhost:5000`

## 🌐 Interfaz Web

La aplicación web permite:
1. **Subir imágenes** mediante drag & drop o selección de archivo
2. **Clasificación automática** usando el modelo entrenado
3. **Visualización de resultados**:
   - Clase predicha con nivel de confianza
   - Probabilidades para cada clase
   - Características extraídas de la imagen

## 🧠 Arquitectura del Modelo

\`\`\`
Entrada (5 características)
    ↓
Capa Oculta 1: 64 neuronas (ReLU) + Dropout (30%)
    ↓
Capa Oculta 2: 32 neuronas (ReLU) + Dropout (20%)
    ↓
Capa Oculta 3: 16 neuronas (ReLU)
    ↓
Capa Salida: 3 neuronas (Softmax)
\`\`\`

**Optimizador:** Adam  
**Función de pérdida:** Sparse Categorical Crossentropy  
**Épocas máximas:** 150 (con early stopping)

## 📊 Características Extraídas

1. **Número de lados**: Detectado mediante aproximación poligonal del contorno
   - Círculo: ~0 (muchos lados → círculo)
   - Cuadrado: 4
   - Triángulo: 3

2. **Proporción ancho/alto**: Relación de aspecto del rectángulo delimitador
   - Círculo/Cuadrado: ~1.0 (figuras simétricas)
   - Triángulo: Variable

3. **Perímetro**: Longitud total del contorno

4. **Área**: Superficie interior del contorno

5. **Compacidad**: Medida de qué tan "redonda" es la figura
   - Círculo: ~1.0 (máxima compacidad)
   - Cuadrado: ~0.785
   - Triángulo: Variable según tipo

## 📁 Estructura del Proyecto

\`\`\`
.
├── scripts/
│   ├── generar_dataset.py         # Generación de imágenes sintéticas
│   ├── extraer_caracteristicas.py # Extracción de características
│   └── entrenar_modelo.py         # Entrenamiento de la red neuronal
├── templates/
│   └── index.html                 # Interfaz web
├── static/
│   └── uploads/                   # Carpeta para imágenes subidas
├── dataset/                       # Dataset generado (creado automáticamente)
│   ├── circulo/
│   ├── cuadrado/
│   └── triangulo/
├── modelo/                        # Modelo entrenado (creado automáticamente)
│   ├── clasificador_figuras.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── clasificador.py                # Módulo para clasificar imágenes nuevas
├── app.py                         # Aplicación web Flask
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
\`\`\`

## 🔍 Uso del Módulo Clasificador

También puedes usar el clasificador directamente en Python:

\`\`\`python
from clasificador import clasificar

# Clasificar una imagen
resultado = clasificar('mi_imagen.png')

print(f"Clase predicha: {resultado['clase_predicha']}")
print(f"Confianza: {resultado['confianza']}")
print(f"Probabilidades: {resultado['probabilidades']}")
\`\`\`

O usar la clase completa:

\`\`\`python
from clasificador import ClasificadorFiguras

# Inicializar clasificador
clf = ClasificadorFiguras()

# Clasificar
resultado = clf.clasificar_imagen('ruta/a/imagen.png')
\`\`\`

## 📈 Métricas de Evaluación

El sistema genera automáticamente:
- **Accuracy** en conjunto de prueba
- **Reporte de clasificación** (precision, recall, f1-score)
- **Matriz de confusión** visualizada
- **Gráficas de entrenamiento** (loss y accuracy por época)

## 🎨 Características de la Interfaz Web

- Diseño moderno con Bootstrap 5
- Drag & drop para subir imágenes
- Vista previa de la imagen
- Animaciones y transiciones suaves
- Responsive (funciona en móviles y tablets)
- Visualización clara de resultados con barras de progreso

## 🛠️ Tecnologías Utilizadas

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, scikit-learn
- **Procesamiento de imágenes**: OpenCV
- **Visualización**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Manipulación de datos**: NumPy, Pandas

## 💡 Funcionamiento General

1. **Generación de datos**: Se crean imágenes sintéticas con OpenCV, variando tamaño, posición, color y fondo para aumentar la robustez del modelo.

2. **Extracción de características**: En lugar de procesar píxeles directamente, extraemos propiedades geométricas que describen la forma (número de lados, área, compacidad, etc.).

3. **Entrenamiento**: Un Perceptrón Multicapa aprende a asociar estas características con cada tipo de figura, ajustando sus pesos mediante backpropagation.

4. **Predicción**: Para una imagen nueva, se extraen sus características, se normalizan, y el modelo predice la clase con mayor probabilidad.

## 🎓 Explicación Técnica

### ¿Por qué características en lugar de píxeles?

- **Eficiencia**: 5 números vs miles de píxeles
- **Interpretabilidad**: Podemos entender qué mide cada característica
- **Generalización**: Funciona con diferentes colores, tamaños y posiciones
- **Rendimiento**: Modelo más pequeño y rápido

### ¿Cómo funciona la extracción?

1. **Binarización**: Convertir imagen a blanco/negro
2. **Detección de contornos**: Encontrar el borde de la figura
3. **Aproximación poligonal**: Simplificar el contorno a líneas rectas
4. **Cálculo de propiedades**: Usar geometría computacional

### ¿Cómo aprende la red neuronal?

1. **Forward pass**: Las características pasan por las capas
2. **Predicción**: La última capa da probabilidades para cada clase
3. **Error**: Se calcula la diferencia con la respuesta correcta
4. **Backpropagation**: Se ajustan los pesos para reducir el error
5. **Iteración**: Se repite con miles de ejemplos

## 🤝 Autor

Proyecto desarrollado por Wilson Avilacomo demostración de clasificación de figuras geométricas usando redes neuronales.

## 📝 Notas

- El modelo está entrenado específicamente para círculos, cuadrados y triángulos simples
- Funciona mejor con figuras bien definidas sobre fondos uniformes
- Las figuras pueden tener cualquier color, pero deben contrastar con el fondo
- El sistema es robusto a variaciones de tamaño y posición
