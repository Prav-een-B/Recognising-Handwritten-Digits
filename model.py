import os
import cv2 
import numpy as np
import tensorflow as tf

'''
this part of code is to build the model


def normalise(dataset):
    dataset = tf.keras.utils.normalize(dataset, axis=1)
    return dataset

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = normalise(x_train)
x_test = normalise(x_test)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 5)

model.save('MYmod.keras') 
'''

'''
this function will preprocess images for better results
'''
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Normalize the image
    img = img / 255.0
    return img
'''
this function will extract the digits using the predict function and argmax function by seeing the maximum among the stimulated nodes to classify the digit
'''
def extract_number(image_path, model, window_size=(28, 28), step_size=28):
    # Preprocess the image
    image = preprocess_image(image_path)
    h, w = image.shape
    prediction = model.predict(window_size)
    digit = np.argmax(prediction)
    return digit
    

# Load the trained model
model = tf.keras.models.load_model('MYmod.keras')

# Path for the image
image_path = 'D:/ROLL_NUMBER/img392.jpg'

# Extract numbers using the model
numbers = extract_number(image_path, model)
print("Extracted numbers:", numbers)
