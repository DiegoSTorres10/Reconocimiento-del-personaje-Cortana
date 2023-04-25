from PIL import Image
import os

carpeta_imagenes = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"

# Recorremos todos los archivos de la carpetaw
for nombre_archivo in os.listdir(carpeta_imagenes):
    # Obtenemos la ruta completa del archivo
    ruta_archivo = os.path.join(carpeta_imagenes, nombre_archivo)
    
    # Verificamos si es un archivo de imagen
    if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Renombramos el archivo agregando "cortana_" al inicio del nombre
        nuevo_nombre = 'cortana_' + nombre_archivo
        nueva_ruta_archivo = os.path.join(carpeta_imagenes, nuevo_nombre)
        os.rename(ruta_archivo, nueva_ruta_archivo)

# for filename in os.listdir(folder_path):
#     if filename.endswith(".webp"):
#         # Abrir la imagen
#         image = Image.open(os.path.join(folder_path, filename))
#         # Guardar la imagen en formato JPEG
#         image = image.convert('RGB')
#         image.save(os.path.join(folder_path, os.path.splitext(filename)[0] + ".jpg"), "JPEG")