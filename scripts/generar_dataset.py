"""
Script para generar un dataset sintético de figuras geométricas.
Crea círculos, cuadrados y triángulos con variaciones en tamaño, posición y color.
"""
import cv2
import numpy as np
import os
from pathlib import Path

# Configuración
TAMAÑO_IMAGEN = 200
CANTIDAD_POR_CLASE = 500
CARPETA_SALIDA = "dataset"

def crear_carpetas():
    """Crea la estructura de carpetas para el dataset"""
    Path(CARPETA_SALIDA).mkdir(exist_ok=True)
    for clase in ['circulo', 'cuadrado', 'triangulo']:
        Path(f"{CARPETA_SALIDA}/{clase}").mkdir(exist_ok=True)

def generar_color_aleatorio():
    """Genera un color RGB aleatorio"""
    return tuple(np.random.randint(50, 255, 3).tolist())

def generar_fondo_aleatorio():
    """Genera un color de fondo aleatorio (tonos claros)"""
    return tuple(np.random.randint(200, 255, 3).tolist())

def generar_circulo(idx):
    """Genera una imagen de un círculo con parámetros aleatorios"""
    img = np.ones((TAMAÑO_IMAGEN, TAMAÑO_IMAGEN, 3), dtype=np.uint8)
    img[:] = generar_fondo_aleatorio()
    
    # Parámetros aleatorios
    radio = np.random.randint(30, 70)
    centro_x = np.random.randint(radio + 10, TAMAÑO_IMAGEN - radio - 10)
    centro_y = np.random.randint(radio + 10, TAMAÑO_IMAGEN - radio - 10)
    color = generar_color_aleatorio()
    
    cv2.circle(img, (centro_x, centro_y), radio, color, -1)
    
    # Añadir algo de ruido
    ruido = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    
    cv2.imwrite(f"{CARPETA_SALIDA}/circulo/circulo_{idx}.png", img)

def generar_cuadrado(idx):
    """Genera una imagen de un cuadrado con parámetros aleatorios"""
    img = np.ones((TAMAÑO_IMAGEN, TAMAÑO_IMAGEN, 3), dtype=np.uint8)
    img[:] = generar_fondo_aleatorio()
    
    # Parámetros aleatorios
    lado = np.random.randint(50, 120)
    x1 = np.random.randint(10, TAMAÑO_IMAGEN - lado - 10)
    y1 = np.random.randint(10, TAMAÑO_IMAGEN - lado - 10)
    color = generar_color_aleatorio()
    
    cv2.rectangle(img, (x1, y1), (x1 + lado, y1 + lado), color, -1)
    
    # Añadir algo de ruido
    ruido = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    
    cv2.imwrite(f"{CARPETA_SALIDA}/cuadrado/cuadrado_{idx}.png", img)

def generar_triangulo(idx):
    """Genera una imagen de un triángulo con parámetros aleatorios"""
    img = np.ones((TAMAÑO_IMAGEN, TAMAÑO_IMAGEN, 3), dtype=np.uint8)
    img[:] = generar_fondo_aleatorio()
    
    # Parámetros aleatorios para un triángulo
    base = np.random.randint(60, 120)
    altura = np.random.randint(60, 120)
    centro_x = np.random.randint(base // 2 + 10, TAMAÑO_IMAGEN - base // 2 - 10)
    centro_y = np.random.randint(altura // 2 + 10, TAMAÑO_IMAGEN - altura // 2 - 10)
    
    # Puntos del triángulo
    punto1 = (centro_x, centro_y - altura // 2)  # Vértice superior
    punto2 = (centro_x - base // 2, centro_y + altura // 2)  # Vértice inferior izquierdo
    punto3 = (centro_x + base // 2, centro_y + altura // 2)  # Vértice inferior derecho
    
    puntos = np.array([punto1, punto2, punto3], dtype=np.int32)
    color = generar_color_aleatorio()
    
    cv2.fillPoly(img, [puntos], color)
    
    # Añadir algo de ruido
    ruido = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    
    cv2.imwrite(f"{CARPETA_SALIDA}/triangulo/triangulo_{idx}.png", img)

def main():
    """Función principal para generar el dataset completo"""
    print("Creando estructura de carpetas...")
    crear_carpetas()
    
    print(f"Generando {CANTIDAD_POR_CLASE} círculos...")
    for i in range(CANTIDAD_POR_CLASE):
        generar_circulo(i)
        if (i + 1) % 100 == 0:
            print(f"  Generados {i + 1}/{CANTIDAD_POR_CLASE}")
    
    print(f"Generando {CANTIDAD_POR_CLASE} cuadrados...")
    for i in range(CANTIDAD_POR_CLASE):
        generar_cuadrado(i)
        if (i + 1) % 100 == 0:
            print(f"  Generados {i + 1}/{CANTIDAD_POR_CLASE}")
    
    print(f"Generando {CANTIDAD_POR_CLASE} triángulos...")
    for i in range(CANTIDAD_POR_CLASE):
        generar_triangulo(i)
        if (i + 1) % 100 == 0:
            print(f"  Generados {i + 1}/{CANTIDAD_POR_CLASE}")
    
    print(f"\n✓ Dataset generado exitosamente en '{CARPETA_SALIDA}/'")
    print(f"  Total de imágenes: {CANTIDAD_POR_CLASE * 3}")

if __name__ == "__main__":
    main()
