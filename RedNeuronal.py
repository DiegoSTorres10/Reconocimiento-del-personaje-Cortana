from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

"""Para convertir las de entrenamiento a 30 X 30  """
# def EstandarizarImagenesEntrenamiento ():
    
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
# def EstandarizarImagenesPrueba ():
#     ruta_original = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba"

#     ruta_estandarizada = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"

#     tamaño = (30, 30)

#     # Itera sobre todas las imágenes en la carpeta original
#     for archivo in os.listdir(ruta_original):
#         # Abre la imagen original
#         imagen_original = Image.open(os.path.join(ruta_original, archivo))

#         # Estandariza el tamaño de la imagen
#         imagen_estandarizada = imagen_original.resize(tamaño)

#         # Guarda la imagen estandarizada en la carpeta correspondiente
#         imagen_estandarizada.save(os.path.join(ruta_estandarizada, archivo))



train_data = []
def TecnicaNormalizadaEntrenamiento ():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_Normalizado"
    file_names = os.listdir(folder_path)
    normalized_folder = "C:/Users/DIDAN/Desktop/RedNeuronal/Normalizacion"

    # Cargar las imágenes y convertirlas en un array NumPy
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path).convert('L')
        imagenes = tf.cast(image, tf.float32)
        imagenes /= 255 #Aqui lo pasa de 0-255 a 0-1
        image_array = tf.keras.preprocessing.image.img_to_array(imagenes)
        normalized_image = tf.keras.preprocessing.image.array_to_img(image_array)
        image_path = os.path.join(normalized_folder, f"{file_name}")
        normalized_image.save(image_path)
        train_data.append([imagenes, int('cortana' in file_name)])
        
test_data=[]
def TecnicaNormalizadaTest ():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"
    file_names = os.listdir(folder_path)
    normalized_folder = "C:/Users/DIDAN/Desktop/RedNeuronal/Normalizacion_prueba"

    # Cargar las imágenes y convertirlas en un array NumPy
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path).convert('L')
        imagenes = tf.cast(image, tf.float32)
        imagenes /= 255 #Aqui lo pasa de 0-255 a 0-1
        image_array = tf.keras.preprocessing.image.img_to_array(imagenes)
        normalized_image = tf.keras.preprocessing.image.array_to_img(image_array)
        image_path = os.path.join(normalized_folder, f"{file_name}")
        normalized_image.save(image_path)
        test_data.append([imagenes, int('cortana' in file_name)])

TecnicaNormalizadaEntrenamiento ()
TecnicaNormalizadaTest ()


# Convertir los datos de entrenamiento y prueba a arrays de NumPy
train_images = np.array([data[0] for data in train_data])
train_labels = np.array([data[1] for data in train_data])

test_images = np.array([data[0] for data in test_data])
test_labels = np.array([data[1] for data in test_data])

# Normalizar los datos
train_images = train_images / 255.0
test_images = test_images / 255.0

# Diseñar la arquitectura de la RNA

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30, 1)),
    tf.keras.layers.Dense(120, activation='tanh'),
    tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(60, activation='tanh'),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(25, activation='tanh'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con un conjunto de validación
history = model.fit(train_images, train_labels, epochs=85, validation_data=(test_images, test_labels))

# Acceder al historial de entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()


# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Accuracy:', test_acc)

# Predecir las etiquetas de las imágenes de prueba
threshold = 0.85
predictions = model.predict(test_images)


# print (predictions, threshold)
predicted_labels = np.where(predictions >= threshold, 1, 0)
# Imprimir las etiquetas predichas para las primeras 10 imágenes de prueba
# print(predictions[:10])

model.save('mi_modelo.h5')