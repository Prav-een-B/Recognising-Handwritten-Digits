Handwritten Digits Recognition

This repository contains a project for recognizing handwritten digits using Python libraries such as TensorFlow, OpenCV, and NumPy.
The model is trained on the MNIST dataset and can predict digits from images.


The model is a simple neural network with the following layers: 

  *Flatten layer to convert the 28x28 pixel images into a 1D array.
  *Two Dense layers with 128 neurons each and ReLU activation.
  *Output Dense layer with 10 neurons (one for each digit) and softmax activation.


The model achieves an accuracy of approximately 98% on the MNIST test dataset.
It can accurately predict handwritten digits from images after proper preprocessing.


To try this make sure you download the the my_mod.keras model and have the model.py all in one directory.
Then you can have your own image to test by changing the path in the model.py
