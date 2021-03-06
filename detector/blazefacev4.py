import cv2
import mediapipe as mp
import random
import numpy as np
from PIL import Image
import face_recognition

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.utils import save_img

from typing import List, Dict

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

### Import the model for predictions
save_path = ('../models/model_cnn')
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
    return image[y1:y2, x1:x2]


#transforming all the detected coordinates in arrays and storing them into a dict
def converting_faces_to_array(image, faces_coordinates):
    encoded_faces = []
    for face in faces_coordinates:
        encoded_faces.append(face_from_coordinates(image, *face))
        #save_img('/frames/', face_from_coordinates(image, *face)[0])
    return encoded_faces

    # encoded_faces = {}
    # for index, face in enumerate(faces_coordinates):
    #     encoded_faces[f"face{index + 1}"] = face_from_coordinates(image, *face)
    # return encoded_faces

def prep_prediction_multi(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img= tf.expand_dims(img,0)
    return img

def prep_prediction_one(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    #img= tf.expand_dims(img,0)
    return img

def pred_mask(encoded_faces) -> List[int]:
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

# #resizing the output to 128 x 128
# def resized_faces_array(encoded_faces):
#     resized_faces_array = {}
#     for key, individual_face in encoded_faces.items():
#         image = Image.fromarray(individual_face)
#         image = image.resize((128,128))
#         resized_faces_array[key] = np.asarray(image)
#     return resized_faces_array


def id_face(small_frame, predicted_faces, encoded_faces):
    # # Load a sample picture and learn how to recognize it.
    face_names = []
    # for face in predicted_faces:
    #     if face == 0:
    #         fernando_image = face_recognition.load_image_file('fernando.jpg')
    #         fernando_face_encoding = face_recognition.face_encodings(fernando_image)[0]

    #         # Load a second sample picture and learn how to recognize it.
    #         emanuel_image = face_recognition.load_image_file("emanuel.png")
    #         emanuel_face_encoding = face_recognition.face_encodings(emanuel_image)[0]

    #         # Create arrays of known face encodings and their names
    #         known_face_encodings = [
    #             fernando_face_encoding,
    #             emanuel_face_encoding
    #         ]
    #         known_face_names = [
    #             "Fernando Jardim",
    #             "Emanuel M??ller"
    #         ]
    #         face_locations = face_recognition.face_locations(small_frame)
    #         face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    #         for face_encoding in face_encodings:
    #             # See if the face is a match for the known face(s)
    #             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    #             name = "Unknown"

    #             # Or instead, use the known face with the smallest distance to the new face
    #             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    #             best_match_index = np.argmin(face_distances)
    #             if matches[best_match_index]:
    #                 name = known_face_names[best_match_index]

    #             face_names.append(name)
    #     else:
    #         face_names.append(0)
    # print(face_names)
    return face_names

def draw_bounding_box(frame, face_names, faces_coordinates, predictions):
    '''
    faces_coordinates : list of lists of integers
    '''
    color_corresp = {
        0: (255,0,0),
        1: (255,165,0),
        2: (0,255,0)
    }
    status_corresp = {
        0: 'no mask',
        1: 'maybe',
        2: 'mask'
    }
    
    # box: x1, y1, x2, y2
    bottom = 4
    left = 4
    right = 4
    
    for index, box in enumerate(faces_coordinates):
               
        #printing boxes
        cv2.rectangle(frame,
                        (box[0] * 4, box[1] * 4),
                        (box[2] * 4, box[3] * 4),
                        color = color_corresp[predictions[index]],
                        thickness = 3)
            
        #printing subtitles
        cv2.rectangle(frame, (box[0], box[3] - 35), (box[2], box[3]), color_corresp[predictions[index]], cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, status_corresp[predictions[index]], (box[0] + 6, box[3]- 6), font, 1.0, (255, 255, 255), 1)
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

            # Only process every 4 frame of video to save time
            if i == 0 or i % 4 == 0:
            #if process_this_frame:

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = face_detection.process(small_frame)

                # Converting faces in coordinates
                faces_coordinates = converting_results_to_coordinates(results, small_frame_shape)

                # Crop the faces as arrays
                encoded_faces = converting_faces_to_array(small_frame, faces_coordinates)
                #print(encoded_faces)
                # Resize
                # resized_faces = resized_faces_array(encoded_faces) # dict {'face1': ndarray,...}

                # Get the prediction for each face (of whether it has a mask)
                #predicted_faces = predict_faces(encoded_faces)
                #if i % 4 == 0:
                predicted_faces = pred_mask(encoded_faces)
                #predicted_faces
                
                face_names = id_face(small_frame, predicted_faces, encoded_faces)

                #process_this_frame = not process_this_frame

            # Update the squares accordingly
            draw_bounding_box(image, face_names, faces_coordinates, predicted_faces)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

run_video()

#to implement:

# Resize frame of video to 1/4 size for faster face recognition processing
# small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# laggy predictions

# Boxes_names

# Identifying_face
