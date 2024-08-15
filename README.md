# Malaria Detection

## Context
Malaria has caused more than 400,000 related death and there were more than 220 million cases happening around the world. To diagnosis if a person has malaria, it is a tedious, time-consuming process in traditional in-person inspection, with affected accuracy due to inter-observer variability. Even worst, the human-inspection can only diagoze Malaria in it later stage, which is too late to take action.

Yet, an automated system can greatly boost efficiency and accuracy of diagonising Malaria in its early stage by using machine learning and artificial intelligence techniques. This system will not only reduce manual error classification but also benefit the human health.

## Objective
This project aims to build an efficient model to detect Malaria. Provided by a red blood cell image, this model should be able to identify if this cell is infected with Malaria or not, and the accuracy of the result should be high enough to build reliability.
<img width="1252" alt="image" src="https://github.com/user-attachments/assets/0e872523-717c-42d2-a6b7-7f39c9f2530e">


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

## Model Developemnt
### Based CNN Model
Using the basic structire to get a quick overview of current performance.
![image](https://github.com/user-attachments/assets/c436ade9-7ab8-4a50-881f-5767bdbeeb16)
![image](https://github.com/user-attachments/assets/fd7d3569-0366-4950-8b73-f7ed4de44cd4)
![image](https://github.com/user-attachments/assets/9af46ad0-c186-408b-955d-f880bdf56f66)

**Observation**

The CNN base model showed promising results, achieving an accuracy of 98%. The confusion matrix indicates high precision and recall, with 1268 true negatives and 1280 true positives. However, there were minor misclassifications: 32 false positives and 20 false negatives. The model accuracy improved steadily during training, demonstrating effective learning. The loss curves indicate minimal overfitting, suggesting good generalization to new data. Overall, the base model provides a strong foundation for malaria detection, balancing simplicity and performance effectively.

Next I will build several other models with few more add on layers and try to check if we can try to improve the model. Therefore try to build a model by adding few layers if required and altering the activation functions.

### 1st Model: adding new layers
Here I am improving the base model by adding another type of layer: Spatial Dropout, since it is useful in convolutional neural networks because it drops entire feature maps rather than individual elements, which can help with overfitting while maintaining spatial coherence.

```python
def build_complex_model(input_shape):
    reset_seeds()  # Ensure reproducibility
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),  # Adding Spatial Dropout
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),  # Adding Spatial Dropout
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),  # Adding Spatial Dropout
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Adjust for one-hot encoded labels
    ])
    return model
```
![image](https://github.com/user-attachments/assets/5a0340d9-5366-4998-80d3-00c63570224f)
![image](https://github.com/user-attachments/assets/9d9ce325-d535-4cb3-8e55-a4b1af06fffb)
![image](https://github.com/user-attachments/assets/c0f50757-80a8-4528-b32a-6ba44dbc708a)

***Observation***

This model, enhanced with Spatial Dropout, achieved a moderate accuracy of 62%. The confusion matrix reveals 718 true negatives and 893 true positives, but also 582 false positives and 407 false negatives, indicating room for improvement. Training and validation accuracy show a steady rise, but validation loss fluctuates, suggesting potential overfitting or sensitivity to training data. The model's performance highlights the challenge of balancing dropout rates to prevent overfitting while maintaining model robustness. Further tuning and additional data preprocessing may be necessary to enhance its accuracy and reliability.

### 2nd Model: adding Batch Normalization and Set LeakyRelu as the activation function
```python
def build_model_with_batch_norm(input_shape):
    reset_seeds()  # Ensure reproducibility
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),  # LeakyReLU activation
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),  # LeakyReLU activation
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(128, (3, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),  # LeakyReLU activation
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(256, (3, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),  # LeakyReLU activation
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        GlobalAveragePooling2D(),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),  # LeakyReLU activation
        Dropout(0.5),
        Dense(2, activation='softmax')  # Output layer with softmax activation
    ])
    return model
```
![image](https://github.com/user-attachments/assets/cdfdde28-c028-4911-8248-f2b5413b50d5)
![image](https://github.com/user-attachments/assets/bad9fd13-39e6-490c-ad52-840b4809148b)
![image](https://github.com/user-attachments/assets/45fee3a8-1e98-40e0-b715-a0afbce88b84)

### 3rd Model: using Data Augmentation

```python
def build_model_with_batch_norm(input_shape):
    reset_seeds()  # Ensure reproducibility
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(64, (3, 3)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(128, (3, 3)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        Conv2D(256, (3, 3)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D((2, 2)),
        SpatialDropout2D(0.3),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model
```
![image](https://github.com/user-attachments/assets/3b121c3f-4fda-4fda-a7a9-8514a3352e3d)
![image](https://github.com/user-attachments/assets/4d268f38-a550-45af-9409-872beaeec62c)
![image](https://github.com/user-attachments/assets/fb353296-f1d2-445b-af30-9ae5cb98bf2a)









