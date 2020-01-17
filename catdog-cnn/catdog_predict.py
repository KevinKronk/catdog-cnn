import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load model
model_file = "..\\catdog-cnn\\catdog.h5"
model = tf.keras.models.load_model(model_file)


# Images to make predictions on
images = [
    "..\\catdog-cnn\\cat_dog\\prediction\\im1.jpg",
    "..\\catdog-cnn\\cat_dog\\prediction\\im2.jpg",
    "..\\catdog-cnn\\cat_dog\\prediction\\im3.jpg",
    "..\\catdog-cnn\\cat_dog\\prediction\\im4.jpg",
    "..\\catdog-cnn\\cat_dog\\prediction\\im5.jpg",
    "..\\catdog-cnn\\cat_dog\\prediction\\im6.jpg"
]


for image in images:
    # Load image and preprocess to the correct size
    im = tf.keras.preprocessing.image.load_img(image, target_size=(64, 64))
    im = tf.keras.preprocessing.image.img_to_array(im)

    # Make prediction and plot image
    plt.imshow(im / 255)
    im = np.expand_dims(im, axis=0)
    im = im / 255
    result = model.predict(im)
    plt.xlabel(f"Is this a dog?: {result[0][0] * 100:.2f}%")
    plt.show()
