### Realtime Hand Gesture Recognition

This repository includes the code and data for Hand Gesture Recognition (0 to 5). I use the OpenCV library to get photos directly from the Camera. Then, this image is through a trained model to predict the outcome.

**Video Demo:** (Click on the image below to view)

[![Click here](http://i3.ytimg.com/vi/c_xwY0vfdcM/maxresdefault.jpg)](https://www.youtube.com/watch?v=c_xwY0vfdcM)

### How to run?

```python
1. Download this repository.
2. Run: python main.py 
```

### Sample of data 
I use the simple finger data set contained in datasets. Some data as shown below.
![](images/signs_data_kiank.png)

To use this data set for the problem, I performed background splitting using the OpenCV library. Data after separation is as follows.

![](images/5.png)

Because the data was too small, I used the TensorFlow library to enhance the data set.

```python
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```

### Train Model 

I use Google Colab to perform this trainning process. Results achieved on Test data sets about 90%.

In this article, I don't use overly complex models, I just use the simpler Model shown below, because my input image is too small (64x64):
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
```
