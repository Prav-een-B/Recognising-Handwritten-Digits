import os
import cv2 
import numpy as np
import tensorflow as tf

'''
this part commented off the code is to build the model


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

# adam optimizer will be used which takes advantage of the AdaGrad and RMSProp

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 5) # running 5 epochs to increase accurracy

model.save('MYmod.keras')  # saving the model 
'''
'''uncomment the above part to build a model'''
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
    # Resize the image to the expected input shape of the model
    image_resized = cv2.resize(image, window_size)
    # Reshape the image to include the batch dimension
    image_resized = image_resized.reshape(1, window_size[0], window_size[1], 1)
    prediction = model.predict(window_size)
    digit = np.argmax(prediction)
    return digit
    

# Load the trained model
model = tf.keras.models.load_model('MYmod.keras')

# Path for the image
image_path = 'img3.jpg'

# Extract numbers using the model
numbers = extract_number(image_path, model)
print("Extracted numbers:", numbers)
