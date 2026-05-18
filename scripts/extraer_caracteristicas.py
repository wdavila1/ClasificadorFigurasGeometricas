"""
Script para extraer características de las figuras geométricas.
Calcula: número de lados, proporción ancho/alto, perímetro, área y compacidad.
"""
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

def extraer_caracteristicas(ruta_imagen):
    """
    Extrae características geométricas de una imagen.
    
    Returns:
        dict: Diccionario con las características extraídas
    """
    # Leer imagen
    img = cv2.imread(ruta_imagen)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para binarizar
    _, binaria = cv2.threshold(gris, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        return None
    
    # Tomar el contorno más grande
    contorno = max(contornos, key=cv2.contourArea)
    
    # Calcular características
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    
    # Evitar divisiones por cero
    if perimetro == 0 or area == 0:
        return None
    
    # Aproximación poligonal para contar lados
    epsilon = 0.04 * perimetro
    aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
    numero_lados = len(aproximacion)
    
    # Limitar número de lados (círculos pueden tener muchos)
    if numero_lados > 10:
        numero_lados = 0  # Consideramos 0 como círculo
    
    # Rectángulo delimitador para calcular proporción
    x, y, ancho, alto = cv2.boundingRect(contorno)
    proporcion_ancho_alto = ancho / alto if alto != 0 else 0
    
    # Compacidad (circularidad)
    compacidad = (4 * np.pi * area) / (perimetro ** 2)
    
    return {
        'numero_lados': numero_lados,
        'proporcion_ancho_alto': proporcion_ancho_alto,
        'perimetro': perimetro,
        'area': area,
        'compacidad': compacidad
    }

def procesar_dataset(carpeta_dataset):
    """
    Procesa todo el dataset y extrae características de cada imagen.
    
    Args:
        carpeta_dataset: Ruta a la carpeta con las imágenes
        
    Returns:
        DataFrame: DataFrame con todas las características y etiquetas
    """
    datos = []
    
    clases = ['circulo', 'cuadrado', 'triangulo']
    
    for clase in clases:
        print(f"Procesando {clase}s...")
        carpeta_clase = os.path.join(carpeta_dataset, clase)
        
        if not os.path.exists(carpeta_clase):
            print(f"  Advertencia: No se encontró la carpeta {carpeta_clase}")
            continue
        
        archivos = [f for f in os.listdir(carpeta_clase) if f.endswith(('.png', '.jpg'))]
        
        for idx, archivo in enumerate(archivos):
            ruta_completa = os.path.join(carpeta_clase, archivo)
            caracteristicas = extraer_caracteristicas(ruta_completa)
            
            if caracteristicas is not None:
                caracteristicas['clase'] = clase
                caracteristicas['ruta_archivo'] = ruta_completa
                datos.append(caracteristicas)
            
            if (idx + 1) % 100 == 0:
                print(f"  Procesados {idx + 1}/{len(archivos)}")
    
    df = pd.DataFrame(datos)
    print(f"\n✓ Características extraídas: {len(df)} imágenes procesadas")
    
    return df

def main():
    """Función principal para extraer características y guardar dataset"""
    carpeta_dataset = "dataset"
    
    if not os.path.exists(carpeta_dataset):
        print(f"Error: No se encontró la carpeta '{carpeta_dataset}'")
        print("Ejecuta primero 'generar_dataset.py'")
        return
    
    # Extraer características
    df = procesar_dataset(carpeta_dataset)
    
    # Guardar dataset procesado
    df.to_csv('caracteristicas_dataset.csv', index=False)
    print(f"✓ Características guardadas en 'caracteristicas_dataset.csv'")
    
    # Mostrar estadísticas
    print("\nEstadísticas del dataset:")
    print(df.groupby('clase').size())
    print("\nPrimeras filas:")
    print(df.head())

if __name__ == "__main__":
    main()
