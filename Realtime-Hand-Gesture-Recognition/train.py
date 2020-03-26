from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from data_processing import *
from keras import layers
from keras import models

### Start: Data Processing

# Data orignal
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

m_train = Y_train_orig.shape[1]
m_test = Y_test_orig.shape[1]

# num_classes = 6
# y_train = np_utils.to_categorical(Y_train_orig, num_classes)
# y_test = np_utils.to_categorical(Y_test_orig, num_classes)
# print("Y_train.shape: ", y_train.shape)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Cup background X_train

X_train = np.zeros(shape=(m_train, 64, 64, 1))
X_test = np.zeros(shape=(m_test, 64, 64, 1))

for i in range(m_train):
    X_train_cup_background[i] = cup_background(X_train_orig[i])
    X_train[i] = cv2.cvtColor(X_train_cup_background[i], cv2.COLOR_BGR2GRAY).reshape(64, 64, 1)

for i in range(m_test):
    X_test_cup_background[i] = cup_background(X_test_orig[i])
    X_test[i] = cv2.cvtColor(X_test_cup_background[i], cv2.COLOR_BGR2GRAY).reshape(64, 64, 1)

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

datagen.fit(X_train)

### End: Data Processing

### Start: View

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

### End: View

### Start: Model

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

### End: Model
### Start: Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 25
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=epochs)

# EPOCHS = 15
# model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=32)

### End: Train Model

### Check Test
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

### Save model and weight
model_json = model.to_json()
with open("model-weight/trainedModel.json", "w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("model-weight/model.h5")
