# Neural Network for Handwritten Digit Recognition

This repository contains a neural network built with TensorFlow from scratch using the MNIST dataset to recognize and classify handwritten digits.

## Overview

The neural network is developed using the TensorFlow library and trained on the MNIST dataset. The goal is to accurately classify handwritten digits into their respective categories (0-9). The repository consists of two main parts:

1. Training the Neural Network: The first part of the code focuses on training the neural network using the MNIST dataset. It includes data preprocessing, setting up the TensorFlow graph, defining the architecture of the neural network, training the model, and evaluating its performance.

2. Loading the Saved Model: The second part of the code demonstrates how to load the saved model and make predictions using new test data. It provides an example of how to use the trained model to classify handwritten digits.

## Dependencies

To run the code in this repository, the following dependencies are required:

- TensorFlow (v1.x)
- NumPy
- PIL (Python Imaging Library)

## Dataset

The MNIST dataset is used for training and evaluating the neural network. It consists of a large number of grayscale images of handwritten digits (28x28 pixels). The dataset is divided into training and testing sets, with corresponding labels indicating the digit in each image.

## Code Structure

The repository contains two Jupyter Notebook files:

1. `train_neural_network.ipynb`: This notebook focuses on training the neural network using TensorFlow. It covers data preprocessing, building the network architecture, training the model, and evaluating its accuracy.

2. `load_saved_model.ipynb`: This notebook demonstrates how to load the saved model obtained from training and use it to make predictions on new test data.

## Instructions

To run the code in this repository, follow these steps:

1. Ensure that all the dependencies mentioned above are installed in your Python environment.

2. Download the MNIST dataset files (`digit_xtrain.csv`, `digit_xtest.csv`, `digit_ytrain.csv`, `digit_ytest.csv`) and place them in the `MNIST` folder.

3. Open and run the `train_neural_network.ipynb` notebook to train the neural network using the MNIST dataset.

4. After training, the model will be saved in the `SavedModel` directory.

5. To make predictions using the saved model, open and run the `load_saved_model.ipynb` notebook. It loads the saved model, loads new test data, and predicts the labels for the test samples.

Please refer to the notebook files for detailed code explanations and comments.

## Results

The trained neural network achieves a high accuracy rate in classifying handwritten digits from the MNIST dataset. The accuracy can be evaluated by running the code provided in the notebooks.

## Conclusion

This project demonstrates the process of building and training a neural network using TensorFlow for handwritten digit recognition. It provides a practical example of how to preprocess data, design a neural network architecture, train the model, and make predictions using a saved model. By understanding and replicating the code, you can gain insights into developing your own deep learning models for image classification tasks.
