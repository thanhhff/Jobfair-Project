from model import *
import cv2
import imutils

# Main function
if __name__ == "__main__":

    # structure of the model
    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # load weights into new model
    model.load_weights("model-weight/model2.h5")
    # model.summary()
    print("\nLoaded model from disk\n")

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 260, 600
    # initialize num of frames
    num_frames = 0
    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        ### Predict;



        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 260, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        # cv2.imshow("Video Feed", clone)
        cv2.imshow("", roi)
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user has pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
