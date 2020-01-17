import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Build Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                            activation=tf.keras.activations.relu, input_shape=(64, 64, 3)))
model.add(tf.keras.layers.Conv2D(48, kernel_size=(3, 3), activation=tf.keras.activations.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                            activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(96, kernel_size=(3, 3), activation=tf.keras.activations.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

model.compile(optimizer=tf.keras.optimizers.Adadelta(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

# Data Augmentation
train_data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_data_aug.flow_from_directory(
    'C:\\Users\\kevkr\\PycharmProjects\\catdog-cnn\\catdog-cnn\\cat_dog\\training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_data_aug.flow_from_directory(
    'C:\\Users\\kevkr\\PycharmProjects\\catdog-cnn\\catdog-cnn\\cat_dog\\test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

from IPython.display import display
from PIL import Image

# Train model with augmented data
model.fit_generator(
    training_set,
    steps_per_epoch=250,
    epochs=10,
    validation_data=test_set,
    validation_steps=150)

print(training_set.class_indices)
model.save("catdog.h5")
print("Saved model.")
