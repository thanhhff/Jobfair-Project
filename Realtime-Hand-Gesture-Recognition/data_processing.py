from resnets_utils import *
import cv2

# Data orignal
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

m_train = Y_train_orig.shape[1]
m_test = Y_test_orig.shape[1]


def cup_background(frame):
    # HSV thresholding: Loại bỏ càng nhiều nền càng tốt
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])

    # Tạo mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    b, g, r = cv2.split(result)
    filter = g.copy()

    ret, mask = cv2.threshold(filter, 10, 255, 1)

    # Ảnh đầu vào nền đen, nổi bật giữa trắng.
    frame[mask == 0] = 0
    frame[mask != 0] = 255

    return frame


X_train_cup_background = X_train_orig.copy()
X_test_cup_background = X_test_orig.copy()

# Cup background X_tran

X_train = np.zeros(shape=(m_train, 64, 64, 1))
X_test = np.zeros(shape=(m_test, 64, 64, 1))

for i in range(m_train):
    X_train_cup_background[i] = cup_background(X_train_orig[i])
    X_train[i] = cv2.cvtColor(X_train_cup_background[i], cv2.COLOR_BGR2GRAY).reshape(64, 64, 1)

for i in range(m_test):
    X_test_cup_background[i] = cup_background(X_test_orig[i])
    X_test[i] = cv2.cvtColor(X_test_cup_background[i], cv2.COLOR_BGR2GRAY).reshape(64, 64, 1)

# print(X_train.shape)
# print(X_train[0].shape)
# print(X_train[0])
# cv2.imwrite("test.png", X_train[0])
