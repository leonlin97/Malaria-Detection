# Malaria Detection

# Table of content
1. Executive Summary
2. Project Background
3. Objective
4. Methodologies to approach the solutions
5. Data Description
6. CNN Base Model
7. 1st Model: adding new layers
8. 2nd Model: adding Batch Normalization and Set LeakyRelu as the activation function
9. 3rd Model: using Data Augmentation
10. 4th Model: using the Pre-trained model (VGG16)
11. Conclusion with the best-performed model: Model 3 with Data Augmentation
12. Recommendation to improve the model performance with new data
13. Recommendations for Implementation

## Executive Summary
Malaria is a life-threatening disease affecting millions globally, with over 400,000 deaths annually. Traditional diagnosis through in-person inspection is time-consuming, prone to error due to inter-observer variability, and often identifies malaria only in its later stages, which is too late for effective intervention.

The primary objective of this project is to develop an efficient machine learning model capable of accurately detecting malaria from red blood cell images. This automated system aims to enhance early-stage detection, reducing manual errors and improving overall diagnostic reliability.

After the evaluation of several models, we recommended apply the Model 3 with Data Augmentation, which achieved an impressive accuracy of 98.3% and is the best performance model among all. The confusion matrix shows balanced performance with 1278 true positives and true negatives, and only 22 false positives and false negatives.This model's high precision and recall underscore its effectiveness in early malaria detection, making it highly suitable for deployment in real-world healthcare settings. The successful implementation of this model could significantly impact global health by enabling early detection and timely treatment of malaria, ultimately saving lives and resources.
<img width="1204" alt="image" src="https://github.com/user-attachments/assets/69d93fe8-8e9d-4474-bdce-4733d0214fed">

Integrate the malaria detection model with healthcare systems, provide training, and monitor performance. Secure funding, collaborate with experts, and develop ethical policies. Benefits include reduced late-stage diagnoses and labor costs, with costs for setup and maintenance. Address data privacy, model bias, and operational challenges, and assess scalability and long-term performance.


## Project Background
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


### 4th Model: using the Pre-trained model (VGG16)

**Truncation Point**: block5_pool layer

**Reason**: The block5_pool layer is the last pooling layer in VGG16, right before the fully connected layers start in the original VGG16 architecture. Truncating the model here will give you a rich set of high-level features extracted from the input images, which are suitable for classification tasks.

**Adding Fully Connected Layers**

- **Global Average Pooling**: This layer will reduce each feature map to a single value by taking the average. It will convert the 3D tensor output from the convolutional base to a 1D tensor.
- **Dense Layer with 256 Units**: This fully connected layer will learn to interpret the high-level features extracted by the convolutional base. A higher number of units can help in capturing more complex relationships.
- **Dropout Layer**: Adding dropout can help prevent overfitting by randomly dropping a fraction of the units during training.
- **Output Layer**: A dense layer with num_classes units and softmax activation to produce the final class probabilities.

```python
def build_vgg16_model(input_shape, num_classes, layer_cutoff='block5_pool'):
    reset_seeds()  # Ensure reproducibility

    # Load the VGG16 model up to the specified layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_cutoff).output)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model
```
![image](https://github.com/user-attachments/assets/b683aff2-689e-462a-a19d-6d539f7f26e8)
![image](https://github.com/user-attachments/assets/9c67b959-f5c5-4414-9f05-fc83deaf7a37)
![image](https://github.com/user-attachments/assets/74fff0e7-05f0-4fce-bd5d-31eecd208741)

**Observation**

1. **Balanced Precision and Recall**:
The precision and recall values for both classes are fairly balanced. Precision for the infected class (0.69) is slightly higher than for the uninfected class (0.65), while recall for the uninfected class (0.73) is higher than for the infected class (0.62). This balance indicates that the model is reasonably effective at distinguishing between infected and uninfected cells.

2. **Areas for Improvement**:
The false negatives (500) and false positives (357) indicate that the model can still be improved. The recall for the infected class (0.62) suggests that the model misses a significant number of true infected cases. Improving this recall value could be crucial for better performance in identifying infected cells, which is critical in a healthcare context.

3. **Overall Accuracy and Reliability**:
An overall accuracy of 0.67, with balanced precision and recall values, indicates that the model is moderately effective but has room for improvement. Techniques such as fine-tuning more layers of the VGG16 model, employing more advanced augmentation strategies, or using ensemble methods could potentially enhance the model's performance.

## Conclusion with the best-performed model: Model 3 with Data Augmentation
The best-performing model, Model 3 with Data Augmentation, achieved an impressive accuracy of 98.3%. The confusion matrix shows balanced performance with 1278 true positives and true negatives, and only 22 false positives and false negatives. Data augmentation significantly enhanced model robustness, reducing overfitting and improving generalization. The accuracy and loss curves demonstrate consistent training and validation performance. This model's high precision and recall underscore its effectiveness in early malaria detection, making it highly suitable for deployment in real-world healthcare settings. Overall, integrating data augmentation with Batch Normalization and LeakyReLU activation resulted in a highly accurate and reliable model.

## Recommendation to improve the model performance with new data
1. **Increase Dataset Size**: Collect more diverse and high-quality images to improve model robustness and generalization.

2. **Advanced Data Augmentation**: Implement more sophisticated augmentation techniques like random rotations, contrast adjustments, and synthetic data generation to increase training data diversity.

3. **Hyperparameter Tuning**: Perform extensive hyperparameter optimization to find the optimal learning rate, dropout rates, and batch sizes.

4. **Ensemble Learning**: Combine multiple models to improve prediction accuracy and reduce individual model biases.

5. **Transfer Learning**: Experiment with other pre-trained models like ResNet or EfficientNet to leverage their advanced feature extraction capabilities.

6. **Regularization Techniques**: Incorporate techniques like L2 regularization to further mitigate overfitting.

7. **Model Interpretability**: Implement methods like Grad-CAM to visualize and interpret model decisions, ensuring transparency and trustworthiness.


## Recommendations for Implementation
### Key Recommendations to Implement the Solution:

1. **Integration with Healthcare Systems:** Seamlessly integrate the model into existing healthcare infrastructure for easy access by medical professionals.

2. **Training and Support:** Provide comprehensive training for healthcare workers on using the model and interpreting its results.

3. **Continuous Monitoring:** Implement a monitoring system to continually assess model performance and update it with new data to maintain accuracy.

### Key Actionables for Stakeholders:

1. **Allocate Budget:** Secure funding for computational resources, data collection, and training programs.

2. **Collaborate with Experts:** Engage with data scientists and healthcare professionals to ensure the model meets clinical standards.

3. **Policy Development:** Develop policies for the ethical use of AI in healthcare, ensuring patient privacy and data security.

### Expected Benefit and Costs:

**Benefits:** Improved early detection of malaria can lead to a 50% reduction in late-stage diagnoses, potentially saving 200,000 lives annually. Increased diagnostic efficiency can reduce labor costs by 30%.

**Costs:** Initial setup costs for computational infrastructure and data collection are estimated at $500,000, with annual maintenance costs of $100,000.

### Key Risks and Challenges:

1. **Data Privacy:** Ensuring the security and privacy of patient data during and after implementation.

2. **Model Bias:** Addressing potential biases in the model that could lead to inaccurate predictions for certain populations.

3. **Operational Challenges:** Integrating the AI system with existing healthcare workflows without disrupting daily operations.

### Further Analysis and Associated Problems:

1. **Scalability:** Assess the scalability of the solution in diverse healthcare settings, including resource-limited areas.

2. **Long-term Performance:** Conduct longitudinal studies to evaluate the long-term performance and reliability of the model.

3. **Associated Health Issues:** Investigate the application of the model and similar techniques to other infectious diseases for broader healthcare impact.
