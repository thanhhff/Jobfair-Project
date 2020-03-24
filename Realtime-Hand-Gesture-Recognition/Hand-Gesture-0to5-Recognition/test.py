# from model import *
# import scipy.misc
# from matplotlib.pyplot import imshow
# from keras.models import model_from_json
# import cv2
# import imutils
#
# bg = None
# global model
#
# model = ResNet50(input_shape=(64, 64, 3), classes=6)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.load_weights("model-weight/model2.h5")
#
# model.summary()
#
# # ########### TEST ############
# # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# #
# # # Normalize image vectors
# # X_train = X_train_orig/255.
# # X_test = X_test_orig/255.
#
# # # Convert training and test labels to one hot matrices
# # Y_train = convert_to_one_hot(Y_train_orig, 6).T
# # Y_test = convert_to_one_hot(Y_test_orig, 6).T
# #
# # print ("number of training examples = " + str(X_train.shape[0]))
# # print ("number of test examples = " + str(X_test.shape[0]))
# # print ("X_train shape: " + str(X_train.shape))
# # print ("Y_train shape: " + str(Y_train.shape))
# # print ("X_test shape: " + str(X_test.shape))
# # print ("Y_test shape: " + str(Y_test.shape))
# # preds = model.evaluate(X_test, Y_test)
# # print ("Loss = " + str(preds[0]))
# # print ("Test Accuracy = " + str(preds[1]))
# # ########### TEST ############
#
# img_path = 'images/3.png'
# img = image.load_img(img_path, target_size=(64, 64))
#
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = x / 255.0
# print('Input image shape:', x.shape)
# my_image = scipy.misc.imread(img_path)
# imshow(my_image)
# import time
# time.sleep(10)
# print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
#
#
# a = model.predict(x)
# print(a)

from resnets_utils import *
import cv2

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

print(X_test_orig[0].shape)

test1 = X_train_orig[0]

# print(X_train_orig)
print(Y_train_orig.shape)

# gray = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
# # gray = cv2.GaussianBlur(gray, (7, 7), 0)
# # global variables
# bg = None
#
# print(type(gray))
#
#
# # --------------------------------------------------
# # To find the running average over the background
# # --------------------------------------------------
# def run_avg(image, aWeight):
#     global bg
#     # initialize the background
#     if bg is None:
#         bg = image.copy().astype("float")
#         return
#
#     # compute weighted average, accumulate it and update the background
#     cv2.accumulateWeighted(image, bg, aWeight)


# # ---------------------------------------------
# # To segment the region of hand in the image
# # ---------------------------------------------
# def segment(image, threshold=30):
#     global bg
#     # find the absolute difference between background and current frame
#     diff = cv2.absdiff(bg.astype("uint8"), image)
#     cv2.imshow("diff = grey - bg", diff)
#     cv2.imshow("grey", image)
#     # threshold the diff image so that we get the foreground
#     thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
#     (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(cnts) == 0:
#         return
#     else:
#         segmented = max(cnts, key=cv2.contourArea)
#         return (thresholded, segmented)
#
#
# cv2.imwrite("a.png", test1)
#
# frame = test1

# # Thực hiện tách nền
# def cup_background(frame):
#     # HSV thresholding: Loại bỏ càng nhiều nền càng tốt
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([0, 0, 120])
#     upper_blue = np.array([180, 38, 255])
#
#     # Tạo mask
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     b, g, r = cv2.split(result)
#     filter = g.copy()
#
#     ret, mask = cv2.threshold(filter, 10, 255, 1)
#
#     # Ảnh đầu vào nền đen, nổi bật giữa trắng.
#     frame[mask == 0] = 0
#     frame[mask != 0] = 255
#
#     return frame
#
#
# X_train_cup_background = cup_background(X_train_orig)
# test = X_train_cup_background[0]
# # print(frame)
# cv2.imwrite("b.png", test)
