import cv2
import PySimpleGUI as sg
from tkinter import filedialog
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image



def procesar_imagen(ruta):
    # Cargar el modelo pre-entrenado
    model = load_model('FiltroDatos.h5')  
    # Cargar la imagen
    imagen = Image.open(ruta)
    imagen.show()

    image = cv2.imread(ruta)
    
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

    prediction = model.predict(np.array([normalized_image]))[0][0]

    # Imprimir el resultado
    if prediction >= 0.65:
        print("La imagen representa a Cortana")
    else:
        print("La imagen no representa a Cortana")


while True:
    print("Menú:")
    print("1) Seleccionar una foto de tu ordenador")
    print("2) Capturar foto desde tu web Camara")
    print("3) Salir")

    opcion = input("Ingresa una opción: ")

    if opcion == "1":
        layout = [[sg.Text("Selecciona una imagen")],
                  [sg.FileBrowse("Buscar", file_types=(("Imagenes", "*.jpg"), ("Imagenes", "*.png")))],
                  [sg.Button("Procesar"), sg.Button("Cancelar")]]
        ventana = sg.Window("Seleccionar imagen").Layout(layout)
        while True:
            evento, valores = ventana.Read()
            if evento == "Procesar":
                archivo_imagen = valores['Buscar']
                ventana.Close()
                procesar_imagen(archivo_imagen)
                break
            elif evento == "Cancelar":
                ventana.Close()
                break

    elif opcion == "2":
        camara = cv2.VideoCapture(0)
        while True:
            _, imagen = camara.read()
            cv2.imshow("Imagen", imagen)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("c"):  # Si se presiona la tecla "c"
                archivo = "captura_camara.jpg"
                cv2.imwrite(archivo, imagen)
                cv2.destroyAllWindows()
                camara.release()
                procesar_imagen(archivo)
                break

    elif opcion == "3":
        break

    else:
        print("Opción inválida. Por favor, intenta de nuevo.")