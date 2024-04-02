# Handwritten Digits Classification

## Problem Overview
In this project, we aim to classify handwritten digits using the MNIST dataset. We will implement three different models for this task: Softmax, Multilayer Perceptron (MLP), and Convolutional Neural Network (CNN) classifiers. The performance of these models will be compared based on accuracy and loss metrics during both training and testing phases.

## Problem Formulation
### Models
- Softmax Classifier
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)

### Model Settings
- Learning rate: 0.01
- Iterations: 1,000
- Dropout rate (when present): 0.2
- Data samples for training: 3,000
- Data samples for validation: 3,000
- Data batch: 200
- Optimizer: Stochastic Gradient Descent
- Activation function: ReLU

## Dataset
We will utilize the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The dataset is widely used for benchmarking image classification algorithms.

## Implementation
Each model will be implemented using Python and a deep learning framework such as TensorFlow or PyTorch. We will train the models using the specified settings and evaluate their performance on both training and testing data.

## Evaluation
The performance of each model will be evaluated based on accuracy and loss metrics. Additionally, we will analyze the computational efficiency and scalability of the models.

## Conclusion
By comparing the performance of Softmax, MLP, and CNN classifiers, we aim to determine the most effective approach for handwritten digit classification using the MNIST dataset. This project will provide insights into the strengths and weaknesses of different deep learning models for image classification tasks.
