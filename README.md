# Malaria Detection

## Context
Malaria has caused more than 400,000 related death and there were more than 220 million cases happening around the world. To diagnosis if a person has malaria, it is a tedious, time-consuming process in traditional in-person inspection, with affected accuracy due to inter-observer variability. Even worst, the human-inspection can only diagoze Malaria in it later stage, which is too late to take action.

Yet, an automated system can greatly boost efficiency and accuracy of diagonising Malaria in its early stage by using machine learning and artificial intelligence techniques. This system will not only reduce manual error classification but also benefit the human health.

## Objective
This project aims to build an efficient model to detect Malaria. Provided by a red blood cell image, this model should be able to identify if this cell is infected with Malaria or not, and the accuracy of the result should be high enough to build reliability.

## Methodologies to approach the solutions
- Data Collection and Preprocessing: Use a comprehensie dataset of red blood cell images, and preprocess the images to ensure they are suitable for training and testing the model. This includes resizing, normalization, and augmentation to enhance the diversity and robustness of the training data.

- Model Development: Experiment with various convolutional neural network (CNN) architectures, including custom models and pre-trained models like VGG16. Implement batch normalization and advanced activation functions (e.g., LeakyReLU) to improve model performance.

- Training and Validation: Split the dataset into training and validation sets, and train the models using appropriate loss functions and optimizers. Employ data augmentation and early stopping to prevent overfitting and improve generalization.

- Evaluation: Evaluate the models using a range of metrics, including accuracy, precision, recall, and F1 score. Use confusion matrices and classification reports to gain insights into the model’s performance and identify areas for improvement.

- Deployment: Plan for the deployment of the final model in a real-world setting, considering factors such as integration with existing healthcare systems, computational requirements, and ethical considerations.

## Data Description
There are a total of 24,958 train and 2,600 test images (colored) that we have taken from microscopic images. These images are of the following categories:<br>


**Parasitized (Infected):** The parasitized cells contain the Plasmodium parasite which causes malaria<br>
**Uninfected:** The uninfected cells are free of the Plasmodium parasites<br>

![image](https://github.com/user-attachments/assets/1db36e1c-7df7-4f2e-a466-cdf027f82546)
