"""
Script para entrenar la red neuronal (Perceptrón Multicapa).
Clasifica figuras geométricas basándose en sus características.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Configurar para resultados reproducibles
np.random.seed(42)
tf.random.set_seed(42)

def cargar_y_preparar_datos(archivo_csv):
    """
    Carga el CSV con características y prepara los datos para entrenamiento.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    print("Cargando datos...")
    df = pd.read_csv(archivo_csv)
    
    # Seleccionar características (X) y etiquetas (y)
    caracteristicas = ['numero_lados', 'proporcion_ancho_alto', 'perimetro', 'area', 'compacidad']
    X = df[caracteristicas].values
    y = df['clase'].values
    
    print(f"Dataset: {len(df)} muestras")
    print(f"Distribución de clases:\n{df['clase'].value_counts()}")
    
    # Codificar etiquetas a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Dividir en entrenamiento y prueba (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalizar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"Conjunto de prueba: {len(X_test)} muestras")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

def crear_modelo(num_caracteristicas, num_clases):
    """
    Crea un Perceptrón Multicapa (MLP) con Keras.
    
    Args:
        num_caracteristicas: Número de características de entrada
        num_clases: Número de clases de salida
        
    Returns:
        modelo: Modelo de Keras compilado
    """
    modelo = keras.Sequential([
        layers.Input(shape=(num_caracteristicas,)),
        layers.Dense(64, activation='relu', name='capa_oculta_1'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', name='capa_oculta_2'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu', name='capa_oculta_3'),
        layers.Dense(num_clases, activation='softmax', name='capa_salida')
    ])
    
    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo y retorna el historial de entrenamiento.
    
    Returns:
        history: Historial de entrenamiento
    """
    print("\nEntrenando modelo...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Entrenar
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluar_modelo(modelo, X_test, y_test, label_encoder):
    """
    Evalúa el modelo y genera métricas y visualizaciones.
    """
    print("\nEvaluando modelo...")
    
    # Predicciones
    y_pred_proba = modelo.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Accuracy en conjunto de prueba: {accuracy:.4f}")
    
    # Reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150)
    print("✓ Matriz de confusión guardada en 'matriz_confusion.png'")
    
    return accuracy

def graficar_historial(history):
    """
    Genera gráficas de loss y accuracy por época.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de Loss
    ax1.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', linewidth=2)
    ax1.set_title('Pérdida (Loss) por Época', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica de Accuracy
    ax2.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    ax2.set_title('Precisión (Accuracy) por Época', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('historial_entrenamiento.png', dpi=150)
    print("✓ Gráficas guardadas en 'historial_entrenamiento.png'")

def guardar_modelo_y_procesadores(modelo, scaler, label_encoder):
    """
    Guarda el modelo entrenado y los procesadores necesarios.
    """
    # Crear carpeta de modelos
    os.makedirs('modelo', exist_ok=True)
    
    # Guardar modelo
    modelo.save('modelo/clasificador_figuras.keras')
    print("✓ Modelo guardado en 'modelo/clasificador_figuras.keras'")
    
    # Guardar scaler y label encoder
    import pickle
    with open('modelo/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('modelo/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ Procesadores guardados en 'modelo/'")

def main():
    """Función principal para entrenar el modelo completo"""
    
    # Verificar que existe el archivo de características
    if not os.path.exists('caracteristicas_dataset.csv'):
        print("Error: No se encontró 'caracteristicas_dataset.csv'")
        print("Ejecuta primero 'extraer_caracteristicas.py'")
        return
    
    # 1. Cargar y preparar datos
    X_train, X_test, y_train, y_test, scaler, label_encoder = cargar_y_preparar_datos(
        'caracteristicas_dataset.csv'
    )
    
    # 2. Crear modelo
    num_caracteristicas = X_train.shape[1]
    num_clases = len(label_encoder.classes_)
    modelo = crear_modelo(num_caracteristicas, num_clases)
    
    print("\nArquitectura del modelo:")
    modelo.summary()
    
    # 3. Entrenar modelo
    history = entrenar_modelo(modelo, X_train, y_train, X_test, y_test)
    
    # 4. Evaluar modelo
    accuracy = evaluar_modelo(modelo, X_test, y_test, label_encoder)
    
    # 5. Generar gráficas
    graficar_historial(history)
    
    # 6. Guardar modelo y procesadores
    guardar_modelo_y_procesadores(modelo, scaler, label_encoder)
    
    print("\n" + "="*50)
    print(f"✓ Entrenamiento completado con éxito!")
    print(f"  Accuracy final: {accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
