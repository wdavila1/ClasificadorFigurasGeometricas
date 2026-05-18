"""
Módulo para clasificar figuras geométricas nuevas usando el modelo entrenado.
"""
import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

class ClasificadorFiguras:
    """Clase para clasificar figuras geométricas"""
    
    def __init__(self, ruta_modelo='modelo'):
        """
        Inicializa el clasificador cargando el modelo y procesadores.
        
        Args:
            ruta_modelo: Carpeta donde están guardados el modelo y procesadores
        """
        self.ruta_modelo = ruta_modelo
        self.modelo = None
        self.scaler = None
        self.label_encoder = None
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Carga el modelo entrenado y los procesadores"""
        try:
            # Cargar modelo
            ruta_modelo_keras = os.path.join(self.ruta_modelo, 'clasificador_figuras.keras')
            self.modelo = tf.keras.models.load_model(ruta_modelo_keras)
            
            # Cargar scaler
            with open(os.path.join(self.ruta_modelo, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Cargar label encoder
            with open(os.path.join(self.ruta_modelo, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✓ Modelo cargado exitosamente")
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            raise
    
    def extraer_caracteristicas(self, imagen):
        """
        Extrae características de una imagen.
        
        Args:
            imagen: Imagen numpy array (BGR)
            
        Returns:
            dict: Características extraídas o None si hay error
        """
        try:
            # Convertir a escala de grises
            if len(imagen.shape) == 3:
                gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gris = imagen
            
            # Binarizar
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
            
            if perimetro == 0 or area == 0:
                return None
            
            # Aproximación poligonal
            epsilon = 0.04 * perimetro
            aproximacion = cv2.approxPolyDP(contorno, epsilon, True)
            numero_lados = len(aproximacion)
            
            # Limitar número de lados
            if numero_lados > 10:
                numero_lados = 0
            
            # Proporción ancho/alto
            x, y, ancho, alto = cv2.boundingRect(contorno)
            proporcion_ancho_alto = ancho / alto if alto != 0 else 0
            
            # Compacidad
            compacidad = (4 * np.pi * area) / (perimetro ** 2)
            
            return {
                'numero_lados': numero_lados,
                'proporcion_ancho_alto': proporcion_ancho_alto,
                'perimetro': perimetro,
                'area': area,
                'compacidad': compacidad
            }
            
        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None
    
    def clasificar_imagen(self, imagen):
        """
        Clasifica una imagen y retorna la predicción con probabilidades.
        
        Args:
            imagen: Imagen numpy array o ruta a archivo
            
        Returns:
            dict: Resultado de la clasificación con clase predicha y probabilidades
        """
        # Si es una ruta, cargar la imagen
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen)
            if imagen is None:
                return {'error': 'No se pudo cargar la imagen'}
        
        # Extraer características
        caracteristicas = self.extraer_caracteristicas(imagen)
        
        if caracteristicas is None:
            return {'error': 'No se pudo extraer características de la imagen'}
        
        # Preparar características para predicción
        X = np.array([[
            caracteristicas['numero_lados'],
            caracteristicas['proporcion_ancho_alto'],
            caracteristicas['perimetro'],
            caracteristicas['area'],
            caracteristicas['compacidad']
        ]])
        
        # Normalizar
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        probabilidades = self.modelo.predict(X_scaled, verbose=0)[0]
        clase_idx = np.argmax(probabilidades)
        clase_predicha = self.label_encoder.inverse_transform([clase_idx])[0]
        
        # Preparar resultado
        resultado = {
            'clase_predicha': clase_predicha,
            'confianza': float(probabilidades[clase_idx]),
            'probabilidades': {
                clase: float(prob) 
                for clase, prob in zip(self.label_encoder.classes_, probabilidades)
            },
            'caracteristicas': caracteristicas
        }
        
        return resultado

# Función auxiliar para uso rápido
def clasificar(ruta_imagen, ruta_modelo='modelo'):
    """
    Función auxiliar para clasificar una imagen rápidamente.
    
    Args:
        ruta_imagen: Ruta a la imagen a clasificar
        ruta_modelo: Carpeta del modelo (default: 'modelo')
        
    Returns:
        dict: Resultado de la clasificación
    """
    clasificador = ClasificadorFiguras(ruta_modelo)
    return clasificador.clasificar_imagen(ruta_imagen)

if __name__ == "__main__":
    # Ejemplo de uso
    print("Ejemplo de uso del clasificador:")
    print("\nfrom clasificador import clasificar")
    print("resultado = clasificar('mi_imagen.png')")
    print("print(resultado)")
