from tensorflow.keras.models import load_model
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2

# Cargar el modelo
# model = load_model('mi_modelo.h5')
# model = load_model('FiltroDatos.h5')


# # Abrir la ventana de diálogo de archivo y permitir que el usuario seleccione una imagen
# root = tk.Tk()
# root.withdraw()
# archivo_imagen = filedialog.askopenfilename()

# Cargar la imagen seleccionada y preprocesarla
# imagen = Image.open("C:/Users/DIDAN/Pictures/perfil.jpg").convert('L')
# imagen = Image.open("C:/Users/DIDAN/Desktop/Prueba.jpg").convert('L')

"""Normalizacion"""
# imagen = Image.open(archivo_imagen).convert('L')
# imagen = np.array(imagen)
# imagen = cv2.resize(imagen, (30, 30))
# imagen = imagen / 255.0
# Hacer la predicción con el modelo y mostrar el resultado al usuario
# prediccion = model.predict(np.array([imagen.reshape((30, 30, 1))]))
# print(prediccion)
# etiqueta_predicha = 'cortana' if prediccion[0][0] >= 0.65 else 'no_cortana'
# print(etiqueta_predicha)

# Función para preprocesar las imágenes de entrenamiento y prueba
def preprocess_image(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ajustar el contraste
    alpha = 1.2 # ajuste de contraste
    beta = 0 # ajuste de brillo
    adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    
    # Redimensionar la imagen a 30x30 píxeles
    resized_image = cv2.resize(adjusted_image, (30, 30))
    
    # Normalizar la imagen
    normalized_image = resized_image / 255.0
    
    return normalized_image


# Cargar el modelo pre-entrenado
model = load_model('FiltroDatos.h5')

# Abrir la ventana de diálogo de archivo y permitir que el usuario seleccione una imagen
root = tk.Tk()
root.withdraw()
archivo_imagen = filedialog.askopenfilename()

# Preprocesar la imagen seleccionada por el usuario
input_data = preprocess_image(archivo_imagen)

# Realizar la predicción
prediction = model.predict(np.array([input_data]))[0][0]

# Imprimir el resultado
if prediction >= 0.65:
    print("La imagen representa a Cortana")
else:
    print("La imagen no representa a Cortana")
