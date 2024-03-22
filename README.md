# Cats vs. Dogs Image Classifier

## Project Overview

This project develops a machine learning model capable of classifying images as either cats or dogs. Utilizing TensorFlow and Keras, I've constructed a Convolutional Neural Network (CNN) that has been trained, validated, and tested on a dataset comprising images of cats and dogs. The goal is to accurately identify an image as belonging to one of the two categories, showcasing the power of deep learning in image classification tasks.

## Dataset

The dataset used in this project is the "Cats and Dogs" dataset, which is publicly available and consists of 3 sets:

- **Training set:** 2,000 images (1,000 cats, 1,000 dogs)
- **Validation set:** 1,000 images (500 cats, 500 dogs)
- **Test set:** 50 images

The images were preprocessed and augmented to improve the robustness and performance of the model. This included rescaling, rotation, width and height shifts, shear transformations, zoom, and horizontal flipping.

## Model Architecture

The CNN model was designed with the following layers:

- Convolutional layers with ReLU activation functions, followed by max-pooling layers
- Dropout layer to reduce overfitting
- A Flatten layer to convert the 2D matrix data to a vector
- Dense layers, including an output layer with a sigmoid activation function to achieve binary classification

The model was compiled with the Adam optimizer and binary cross-entropy loss function, aiming for high accuracy in distinguishing between cats and dogs.

## Training and Results

The model underwent training for 15 epochs with early stopping implemented to prevent overfitting. We achieved promising results, demonstrating the model's ability to learn and differentiate between cats and dogs with a high degree of accuracy.

- **Training Accuracy:** Reached up to approximately 73.77%
- **Validation Accuracy:** Peaked around 72.32%

## Conclusion and Future Work

This project highlights the effectiveness of CNNs in image classification tasks. Future work could explore more sophisticated models, further data augmentation techniques, or the use of transfer learning to enhance performance and accuracy.
