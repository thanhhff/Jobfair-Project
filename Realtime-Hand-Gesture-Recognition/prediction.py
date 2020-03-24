from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import scipy.misc
import cv2

### Load the structure of the model
json_file = open('model-weight/trainedModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
from matplotlib.pyplot import imshow

# load weights into new model
loaded_model.load_weights("model-weight/model2.h5")
print("\nLoaded model from disk\n")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ########### TEST ############

img_path = 'images/5.png'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
x = x.reshape(64, 64, 1)
x = np.expand_dims(x, axis=0)

x = x / 255.0

print('Input image shape:', x.shape)
my_image = scipy.misc.imread(img_path)
imshow(my_image)

print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")

print(loaded_model.predict(x).argmax())
