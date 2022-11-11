from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import time
import cv2 as cv
import os



# load our serialized face detector model from disk
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model('mask_detector.model')



def detect_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob from it

    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()


	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

	# loop over the detections
    for i in range(detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection

        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startx, starty, endx, endy = box.astype('int')

            # ensure the bounding boxes fall within the dimensions of the frame
            startx, starty = max(0, startx), max(0, starty)
            endx, endy = min(w-1, endx), min(h-1, endy)

            # extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            face = frame[starty:endy, startx:endx]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startx, starty, endx, endy))

    # only make a predictions if at least one face was detected
    if len(faces)>0:
        # for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        faces = np.array(faces, dtype='float32')
        preds = maskNet.predict(faces, batch_size=32, verbose=0)

    # return a 2-tuple of the face locations and their corresponding locations
    return locs, preds



### video capture
cap = cv.VideoCapture(0)

### to save the video
fourcc = cv.VideoWriter_fourcc(*'MP4V')
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
writer = cv.VideoWriter('mask_detection.mp4', fourcc, 30, (width, height))

# loop over the frames from the video stream
while cap.isOpened():

    ret, frame = cap.read()
    if ret:

        ## detect faces in the frame and determine if they are wearing a face mask or not
        locs, preds = detect_mask(frame, faceNet, maskNet)

        ## loop over the detected face locations and their corresponding locations
        for box, pred in zip(locs, preds):
            ## unpack the bounding box and predictions
            startx, starty, endx, endy = box
            mask, withoutMask = pred

            ## determine the class label and color we'll use to draw the bounding box and text
            label = 'Mask' if mask > withoutMask else 'No Mask'
            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

            ## include the probability in the label
            label = '{}: {:.2f}%'.format(label, max(mask, withoutMask) * 100)

            ### display the label and bounding box rectangle on the output frame
            cv.putText(frame, label, (startx, starty-10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv.rectangle(frame, (startx, starty), (endx, endy), color, 2)
        writer.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) ==27:
            break
    else:
        break
writer.release()
cap.release()
cv.destroyAllWindows()