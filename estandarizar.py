from PIL import Image
import os


"""Para convertir las de entrenamiento a 30 X 30  """

# ruta_original = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_SN"

# ruta_estandarizada = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_Normalizado"

# tamaño = (30, 30)

# # Itera sobre todas las imágenes en la carpeta original
# for archivo in os.listdir(ruta_original):
#     imagen_original = Image.open(os.path.join(ruta_original, archivo))

#     # Estandariza el tamaño de la imagen
#     imagen_estandarizada = imagen_original.resize(tamaño)

#     # Guarda la imagen estandarizada en la carpeta correspondiente
#     imagen_estandarizada.save(os.path.join(ruta_estandarizada, archivo))


"""Para normalizar las imagenes de prueba"""

ruta_original = "C:/Users/DIDAN/Desktop/RedNeuronal/borrar"

ruta_estandarizada = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"

tamaño = (30, 30)

# Itera sobre todas las imágenes en la carpeta original
for archivo in os.listdir(ruta_original):
    # Abre la imagen original
    imagen_original = Image.open(os.path.join(ruta_original, archivo))

    # Estandariza el tamaño de la imagen
    imagen_estandarizada = imagen_original.resize(tamaño)

    # Guarda la imagen estandarizada en la carpeta correspondiente
    imagen_estandarizada.save(os.path.join(ruta_estandarizada, archivo))