# Real-time Age, Gender, and Ethnicity Prediction from Facial Images

Age, Gender, and Ethnicity Prediction from Facial Images using CNN
This project employs Convolutional Neural Networks (CNNs) within a Sequential model to achieve real-time predictions of age, gender, and ethnicity from facial images. The project utilizes the 'age-gender-and-ethnicity face data' dataset from Kaggle and integrates OpenCV (cv2) for webcam-based real-time model inference.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Exploration](#data-exploration)
4. [Model Training](#model-training)
5. [Conclusion](#conclusion)

## Introduction
Facial recognition has become a pivotal aspect of computer vision applications, and predicting demographic information adds another layer of understanding. This project aims to leverage CNNs to predict age, gender, and ethnicity in real-time, contributing to the advancement of facial recognition technologies.

## Dataset
The project utilizes the 'age-gender-and-ethnicity face data' dataset from Kaggle, containing facial data with information on 104 different ages, 5 ethnicities, and gender. 

## Data Exploration
To visualize the data and examine the distribution of classes, pixel values are converted to a Numpy array.
1. Converting Pixels to Numpy Array:

```python
num_pixels = len(df['pixels'][0].split(" "))
img_height = int(np.sqrt(len(df['pixels'][0].split(" "))))
img_width = int(np.sqrt(len(df['pixels'][0].split(" "))))
print(num_pixels, img_height, img_width)
```
Information necessary to convert pixel values in the 'pixels' column of the dataframe to numbers is obtained. The number of pixels in each image (num_pixels), the height of the image (img_height), and the width of the image (img_width) are calculated.

2.Converting Pixels to Float32 Type:
```python
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
```
Each pixel array in the 'pixels' column is parsed from string format, split, and then converted to the float32 data type. This process is applied to the entire 'pixels' column. The age, ethnicity, and gender distribution is visualized.

![images0](https://github.com/elifsare/Real-time-Age-Gender-Ethnicity-Detection/blob/41825ecfdc4d19de7e3b19bf134e0c0507d23a30/hist.png)


## Model Training

The GenderAgeEthnicityModel class is used to create and manage models for predicting age, gender, or ethnicity from images. It provides a convenient interface for training, evaluating, plotting training history, and saving the best model.

* The build_model method is the class's model creation method. Depending on the model_type parameter, it creates different models for predicting age, gender, or ethnicity.
```python
def build_model(self):
    model = Sequential()
    if self.model_type == 'age':
        # ... Specify age prediction model architecture
    elif self.model_type == 'gender':
        # ... Specify gender prediction model architecture
    elif self.model_type == 'ethnicity':
        # ... Specify ethnicity prediction model architecture
    else:
        raise ValueError("Invalid model type. Supported types are 'age', 'gender', and 'ethnicity'.")
    return model
```
* The train_model method is used to train the model. It initiates training with the specified number of epochs and a callback list, returning the training history.
* The evaluate_model method evaluates the trained model on both the training and validation datasets, returning the loss and accuracy values.
* The plot_history method visualizes training and testing data.
* The save_model_checkpoint method creates a ModelCheckpoint object to save the best model during training and returns this object.

## Conclusion
Real-time predictions are facilitated by a Haarcascade classifier for face recognition. The project integrates pre-trained CNN models for age, gender, and ethnicity predictions, creating a comprehensive solution. Continuous improvements and optimizations can further enhance the accuracy and efficiency of the model.

Feel free to explore, modify, and contribute to the project for the advancement of facial recognition capabilities.


