"""
Aplicación web Flask para clasificar figuras geométricas.
Permite subir imágenes y obtener predicciones del modelo.
"""
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from clasificador import ClasificadorFiguras

# Configuración de Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB máximo
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Crear carpeta de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inicializar clasificador
try:
    clasificador = ClasificadorFiguras()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    clasificador = None

def extension_permitida(filename):
    """Verifica si la extensión del archivo es permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar una imagen subida"""
    
    # Verificar que se subió un archivo
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se subió ninguna imagen'}), 400
    
    archivo = request.files['imagen']
    
    # Verificar que el archivo tiene nombre
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    # Verificar extensión
    if not extension_permitida(archivo.filename):
        return jsonify({'error': 'Formato de archivo no permitido. Use PNG, JPG o JPEG'}), 400
    
    # Verificar que el clasificador esté disponible
    if clasificador is None:
        return jsonify({'error': 'El modelo no está disponible. Entrénalo primero.'}), 500
    
    try:
        # Leer la imagen directamente desde memoria
        archivo_bytes = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(archivo_bytes, cv2.IMREAD_COLOR)
        
        if imagen is None:
            return jsonify({'error': 'No se pudo procesar la imagen'}), 400
        
        # Clasificar
        resultado = clasificador.clasificar_imagen(imagen)
        
        # Verificar si hubo error
        if 'error' in resultado:
            return jsonify(resultado), 400
        
        # Redondear valores para mejor visualización
        resultado['confianza'] = round(resultado['confianza'] * 100, 2)
        for clase in resultado['probabilidades']:
            resultado['probabilidades'][clase] = round(
                resultado['probabilidades'][clase] * 100, 2
            )
        
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@app.route('/salud')
def salud():
    """Endpoint para verificar el estado de la aplicación"""
    estado = {
        'aplicacion': 'funcionando',
        'modelo_cargado': clasificador is not None
    }
    return jsonify(estado)

if __name__ == '__main__':
    # Verificar que el modelo existe
    if not os.path.exists('modelo/clasificador_figuras.keras'):
        print("\n" + "="*60)
        print("⚠️  ADVERTENCIA: No se encontró el modelo entrenado")
        print("="*60)
        print("\nPara entrenar el modelo, ejecuta estos comandos:")
        print("  1. python scripts/generar_dataset.py")
        print("  2. python scripts/extraer_caracteristicas.py")
        print("  3. python scripts/entrenar_modelo.py")
        print("\nLuego reinicia la aplicación.")
        print("="*60 + "\n")
    
    # Iniciar servidor
    app.run(debug=True, host='0.0.0.0', port=5000)
