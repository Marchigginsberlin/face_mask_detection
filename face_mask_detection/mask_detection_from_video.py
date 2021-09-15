import cv2
import mediapipe as mp
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import save_img

from typing import List, Dict


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

### Import the model for predictions
save_path = ('../models/model_cnn_2_nd_mp2')
model = tf.keras.models.load_model(save_path)

#transforming coordinates of the face into a cropped image
def converting_results_to_coordinates(results, image_shape):
    faces_coordinates = []
    if results.detections is not None:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x1 = max(box.xmin * image_shape[1], 1)
            y1 = max(box.ymin * image_shape[0], 1)
            x2 = min(x1 + box.width * image_shape[1], image_shape[1])
            y2 = min(y1 + box.height * image_shape[0], image_shape[0])
            faces_coordinates.append([int(x1), int(y1), int(x2), int(y2)])
            print(faces_coordinates[-1])
    return faces_coordinates

#transforming coordinates of the face into a cropped image
def face_from_coordinates(image,x1,y1,x2,y2):
    if y1 < 12 or x1 < 6:
        return image[(y1):(y2), (x1):(x2)]
    return image[(y1-10):(y2+10), (x1-5):(x2+5)]


#transforming all the detected coordinates in arrays and storing them into a dict
def converting_faces_to_array(image, faces_coordinates):
    encoded_faces = []
    for face in faces_coordinates:
        encoded_faces.append(face_from_coordinates(image, *face))
        #save_img('/frames/', face_from_coordinates(image, *face)[0])
    return encoded_faces

# preprocess cropped image for model prediction
def prep_prediction_multi(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img= tf.expand_dims(img,0)
    img = img / 255
    return img

# preprocess cropped image for model prediction
def prep_prediction_one(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img = img / 255
    #img= tf.expand_dims(img,0)
    return img

def pred_mask(encoded_faces: Dict) -> List[int]:
    if len(encoded_faces) == 1:
        face = prep_prediction_one(encoded_faces)
        print(model.predict(face))
        return [model.predict(face).argmax()]
    elif len(encoded_faces) > 1:
        faces = []
        for j in range(len(encoded_faces)):
            face = prep_prediction_multi(encoded_faces[j])
            faces.append(face)
        predictions = []
        for face in faces:
            predictions.append(model.predict(face).argmax())
        return predictions
    else:
        return []




def draw_bounding_box(frame, faces_coordinates, predictions):
    '''
    faces_coordinates : list of lists of integers
    '''
    color_corresp = {
        0: (255,0,0),
        1: (255,165,0),
        2: (0,255,0)
    }

    status_corresp = {
        0 : 'no mask',
        1 : 'bad_mask',
        2 : 'mask'
    }

    assert len(faces_coordinates) == len(predictions), 'Coord different than predictions.'


    for index, box in enumerate(faces_coordinates):
        # box: x1, y1, x2, y2
        bottom = 4
        left = 4
        right = 4
        cv2.rectangle(frame,
                        (box[0] * 4, box[1]* 4),
                        (box[2] * 4, box[3] *4 ),
                        color = color_corresp[predictions[index]],
                        thickness = 3)
        cv2.rectangle(frame, (box[0]*4, box[3] * 4 - 35), (box[2] * 4, box[3] *4), color_corresp[predictions[index]], cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, status_corresp[predictions[index]], (box[0] * 4 + 6, box[3] *4 - 6), font, 1.0, (255, 255, 255), 1)
        print(cv2.putText)


def run_video():
# For webcam input:
    cap = cv2.VideoCapture(0)
    success, image = cap.read()
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    image_shape = image.shape
    small_frame_shape =  small_frame.shape
    print(image_shape)
    process_this_frame = True
    i = 0

    with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

            # Only process every other frame of video to save time
            if process_this_frame:

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_detection.process(small_frame)

                # Converting faces in coordinates
                faces_coordinates = converting_results_to_coordinates(results, small_frame_shape)

                # Crop the faces as arrays
                encoded_faces = converting_faces_to_array(small_frame, faces_coordinates)



                # Get the prediction for each face (of whether it has a mask)
                predicted_faces = pred_mask(encoded_faces)


            i += 1
            process_this_frame = not process_this_frame

            # Update the squares accordingly
            draw_bounding_box(image, faces_coordinates, predicted_faces)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

run_video()
