Name:Saurav Yadav Comapny: CODTECH IT SOLUTIONS ID:CT08DS590 Domain: Data Science Duration: 12Dec2024 - 12 Jan2025



<img width="1280" alt="Screenshot 2025-01-03 at 6 42 41 PM" src="https://github.com/user-attachments/assets/6c14bea3-3355-48df-b081-a391398e6122" />




Project Overview: Deep Learning Model for Image Classification using Convolutional Neural Networks (CNN) with TensorFlow
This project implements a Convolutional Neural Network (CNN) for Image Classification on the CIFAR-10 dataset using TensorFlow. The model is designed to classify images from ten different categories, showcasing the process of building, training, and evaluating a CNN model. It also includes visualizations of the model’s performance, such as training and validation accuracy and loss curves, as well as predictions on random test images.

Problem Statement:
The objective of this project is to train a CNN model for image classification on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes. The model will be trained using TensorFlow and will be evaluated using test data. The deliverable includes a functional model along with visualizations of the results, such as accuracy and loss plots, and the classification of sample test images.

Steps to Implement:
Dataset Loading and Preprocessing:

Load the CIFAR-10 dataset using TensorFlow’s built-in functionality.
Normalize the image pixel values to a range of [0, 1] by dividing by 255.
Convert the labels to one-hot encoding for multi-class classification.
Model Architecture:

Define a Convolutional Neural Network (CNN) using TensorFlow Keras.
The CNN architecture consists of:
3 Convolutional layers with ReLU activation and MaxPooling.
Flattening layer to convert the 2D feature maps into a 1D vector.
Dense layers for classification, including a final softmax layer for multi-class output.
Model Compilation:

Compile the model with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
Model Training:

Train the model for 10 epochs with a validation split of 20% to evaluate the model’s performance during training.
Use a batch size of 64 for processing the data.
Model Evaluation:

Evaluate the model on the test set to determine the final accuracy.
Visualization:

Plot the training and validation accuracy over epochs.
Plot the training and validation loss over epochs.
Predict and visualize results by displaying 10 random test images along with their true labels and predicted labels.
