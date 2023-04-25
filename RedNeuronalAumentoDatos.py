from PIL import Image
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import shutil

borrar1 = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_aumentado"
borrar2 = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_aumentado"
# eliminar el contenido de la carpeta
shutil.rmtree(borrar1)
os.mkdir(borrar1)

shutil.rmtree(borrar2)
os.mkdir(borrar2)


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = []
def TecnicaAumentoEntrenamiento():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_Normalizado"
    image_files = os.listdir(folder_path)
    normalized_folder = "C:/Users/DIDAN/Desktop/RedNeuronal/Entrenamiento_aumentado"

    # Usar el generador de aumento de datos para obtener imágenes adicionales
    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        image = load_img(image_path, target_size=(30, 30))
        image = image.convert('L')
        image_array = img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape)
        i = 0
        for batch in datagen.flow(image_array, batch_size=1,
                                save_to_dir=normalized_folder, 
                                save_prefix=image_name[:-4], save_format='jpeg'):
            i += 1
            if i >= 5: # Generar 5 imágenes adicionales para cada imagen original
                break
            # Cargar la imagen aumentada y guardarla en el array
            aug_image_files = os.listdir(normalized_folder)
            for aug_image_name in aug_image_files:
                if aug_image_name.startswith(image_name[:-4]) and aug_image_name != image_name:
                    aug_image_path = os.path.join(normalized_folder, aug_image_name)
                    aug_image = load_img(aug_image_path, target_size=(30, 30))
                    aug_image_array = img_to_array(aug_image)
                    train_data.append([aug_image_array, int('cortana' in image_name)])




test_data=[]
def TecnicaAumentoPrueba():
    folder_path = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_Normalizado"
    image_files = os.listdir(folder_path)
    normalized_folder = "C:/Users/DIDAN/Desktop/RedNeuronal/Prueba_aumentado"

    # Usar el generador de aumento de datos para obtener imágenes adicionales
    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        image = load_img(image_path, target_size=(30, 30))
        image = image.convert('L')
        image_array = img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape)
        i = 0
        for batch in datagen.flow(image_array, batch_size=1,
                                save_to_dir=normalized_folder, 
                                save_prefix=image_name[:-4], save_format='jpeg'):
            i += 1
            if i >= 5: # Generar 5 imágenes adicionales para cada imagen original
                break
            # Cargar la imagen aumentada y guardarla en el array
            aug_image_files = os.listdir(normalized_folder)
            for aug_image_name in aug_image_files:
                if aug_image_name.startswith(image_name[:-4]) and aug_image_name != image_name:
                    aug_image_path = os.path.join(normalized_folder, aug_image_name)
                    aug_image = load_img(aug_image_path, target_size=(30, 30))
                    aug_image_array = img_to_array(aug_image)
                    test_data.append([aug_image_array, int('cortana' in image_name)])

TecnicaAumentoEntrenamiento ()
TecnicaAumentoPrueba()


# Convertir los datos de entrenamiento y prueba a arrays de NumPy
train_images = np.array([data[0] for data in train_data])
train_labels = np.array([data[1] for data in train_data])


test_images = np.array([data[0] for data in test_data])
test_labels = np.array([data[1] for data in test_data])



# Diseñar la arquitectura de la RNA

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(30, 30, 3)),
    tf.keras.layers.Dense(520, activation='tanh'),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(120, activation='tanh'),
    tf.keras.layers.Dense(60, activation='tanh'),

    tf.keras.layers.Dense(30, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con un conjunto de validación
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

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

model.save('AumentoDatos.h5')