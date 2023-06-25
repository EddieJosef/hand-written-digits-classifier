# Handwritten Digit Recognition using Neural Network

This repository contains code for building a neural network from scratch using TensorFlow to recognize and classify handwritten digits from the MNIST dataset.

## Dependencies

- TensorFlow (version 1.x)
- NumPy
- PIL (Python Imaging Library)

## Dataset

The MNIST dataset is used for training and testing the neural network. The dataset consists of images of handwritten digits from 0 to 9.

- Training images: `MNIST/digit_xtrain.csv`
- Training labels: `MNIST/digit_ytrain.csv`
- Test images: `MNIST/digit_xtest.csv`
- Test labels: `MNIST/digit_ytest.csv`

## Data Preprocessing

- The pixel values of the images are rescaled between 0 and 1 by dividing them by 255.
- The target values (labels) are converted to one-hot encoding.

## Neural Network Architecture

The neural network architecture consists of two hidden layers and an output layer.

- Hyperparameters:
  - Number of epochs: 50
  - Learning rate: 1e-4
  - Number of neurons in the first hidden layer: 512
  - Number of neurons in the second hidden layer: 64

## Training

The training process involves the following steps:

1. Batching the training data into smaller batches.
2. Forward propagation through the network.
3. Calculating the loss using softmax cross-entropy.
4. Updating the network weights using the Adam optimizer.
5. Calculating the accuracy on the training set and logging it.

## Validation

During the training process, a validation dataset is used to monitor the model's performance. The accuracy and loss on the validation set are logged for each epoch.

## Testing and Evaluation

After training, the model is tested on the test dataset to evaluate its performance. The accuracy on the test set is calculated and displayed as a percentage.

## Saving the Model

The trained model is saved using the SavedModel format.

## Making Predictions

The model can be used to make predictions on new handwritten digits. An example image is provided (`MNIST/test_img.png`), which is preprocessed and fed into the model for prediction.

## TensorBoard Integration

TensorBoard is used to visualize the training process and monitor the model's performance. The following information is logged:

- Accuracy and loss metrics
- Histograms of weights and biases
- Input images

## Usage

To run the code, make sure you have the required dependencies installed. Then, execute the script and observe the training progress and evaluation results.

```shell
python neural_network_mnist.py
```

Note: This code is compatible with TensorFlow 1.x.

## Author

Created by [Your Name]

Feel free to contribute to this repository and experiment with different architectures or datasets.
