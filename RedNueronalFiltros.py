from PIL import Image
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
import shutil

borrar1 = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_filtros"
borrar2 = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_filtros"
# eliminar el contenido de la carpeta
shutil.rmtree(borrar1)
os.mkdir(borrar1)

shutil.rmtree(borrar2)
os.mkdir(borrar2)




train_data = []
def TecnicaFiltroEntrenamiento():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_Normalizado"
    file_names  = os.listdir(folder_path)
    processed_folder  = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_filtros"

    # Cargar las imágenes y aplicar filtros y retoques
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        
        # Aplicar filtro de blanco y negro
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar retoque de contraste
        alpha = 1.2 # ajuste de contraste
        beta = 0 # ajuste de brillo
        adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
        
        # Guardar la imagen procesada
        processed_image_path = os.path.join(processed_folder, f"{file_name}")
        cv2.imwrite(processed_image_path, adjusted_image)
        
        # Agregar la imagen al arreglo de datos de prueba
        test_data.append([adjusted_image, int('cortana' in file_name)])



test_data=[]
def TecnicaFiltroPrueba():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"
    file_names  = os.listdir(folder_path)
    processed_folder  = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_filtros"

    # Cargar las imágenes y aplicar filtros y retoques
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        
        # Aplicar filtro de blanco y negro
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar retoque de contraste
        alpha = 1.2 # ajuste de contraste
        beta = 0 # ajuste de brillo
        adjusted_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
        
        # Guardar la imagen procesada
        processed_image_path = os.path.join(processed_folder, f"{file_name}")
        cv2.imwrite(processed_image_path, adjusted_image)
        
        # Agregar la imagen al arreglo de datos de prueba
        train_data.append([adjusted_image, int('cortana' in file_name)])


TecnicaFiltroEntrenamiento ()
TecnicaFiltroPrueba()


# Convertir los datos de entrenamiento y prueba a arrays de NumPy
train_images = np.array([data[0] for data in train_data])
train_labels = np.array([data[1] for data in train_data])

test_images = np.array([data[0] for data in test_data])
test_labels = np.array([data[1] for data in test_data])


# preprocesar datos
train_images = train_images / 255.0
test_images = test_images / 255.0


# # compilar modelo
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(30, 30, 1)),
#     tf.keras.layers.Dense(60, activation='relu'),
#     tf.keras.layers.Dropout(0.10),
#     tf.keras.layers.Dense(30, activation='relu'),
#     tf.keras.layers.Dropout(0.9),
#     tf.keras.layers.Dense(25, activation='relu'),
#     tf.keras.layers.Dropout(0.8),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# compilar modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# entrenar modelo
history = model.fit(train_images, train_labels, epochs=85, validation_data=(test_images, test_labels))

# Acceder al historial de entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()
# Evalúa el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Accuracy:', test_acc)



# Predecir las etiquetas de las imágenes de prueba
threshold = 0.85
predictions = model.predict(test_images)


# print (predictions, threshold)
predicted_labels = np.where(predictions >= threshold, 1, 0)
# Imprimir las etiquetas predichas para las primeras 10 imágenes de prueba
# print(predictions[:10])

model.save('FiltroDatos.h5')