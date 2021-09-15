import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from PIL import Image



###load a picture
def load_picture(path_to_file):
    image = cv2.imread(path_to_file)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    return image

### initalize face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection= mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_faces(image):
    results = face_detection.process(image)
    return results

### getting cooridnates of detected faces from the image

def converting_results_to_coordinates(results, image_shape):
    faces_coordinates = []
    if results.detections is not None:
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x1 = max((box.xmin) * image_shape[1], 1)
            y1 = max((box.ymin) * image_shape[0], 1)
            x2 = min(x1 + box.width * image_shape[1], image_shape[1])
            y2 = min(y1 + box.height * image_shape[0], image_shape[0])
            faces_coordinates.append([int(x1), int(y1), int(x2), int(y2)])
            #print(faces_coordinates[-1])
    return faces_coordinates

### transforming coordinates of the face into a cropped image
def face_from_coordinates(image,x1,y1,x2,y2):
    if y1 < 12 or x1 < 6:
        return image[(y1):(y2), (x1):(x2)]
    return image[(y1-10):(y2+10), (x1-5):(x2+5)]

def converting_faces_to_array(image, faces_coordinates):
    encoded_faces = []
    for face in faces_coordinates:
        encoded_faces.append(face_from_coordinates(image, *face))
        #save_img('/frames/',
    return encoded_faces

### preprocessing image for prediction

def prep_prediction_multi(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img= tf.expand_dims(img,0)
    img = img / 255
    return img

def prep_prediction_one(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img = img / 255
    #img= tf.expand_dims(img,0)
    return img

### predicting no mask, wrong mask or mask for each face

def pred_mask(encoded_faces):
    if len(encoded_faces) == 1:
        face = prep_prediction_one(encoded_faces)
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
        return [3]

### draw the image

def draw_image(frame, faces_coordinates, predictions):
    '''
    faces_coordinates : list of lists of integers
    '''
    color_corresp = {
        0: (255,0,0),
        1: (255,165,0),
        2: (0,255,0)
    }

    assert len(faces_coordinates) == len(predictions), 'Coord different than predictions.'

    for index, box in enumerate(faces_coordinates):
        # box: x1, y1, x2, y2
        cv2.rectangle(frame,
                        ((box[0]), (box[1])),
                        ((box[2]), (box[3])),
                        color_corresp[predictions[index]],3)

    #webbrowser.open(frame)
    img = Image.fromarray(frame, 'RGB')
    #img.save('analyzed_image.png')
    img.show()

#### Run the follwing lines of code to analyze your image

model_path = ('../models/model_cnn_2_nd_mp2')
path_to_file = '../raw_data/archive/images/maksssksksss6.png'

model = tf.keras.models.load_model(model_path)

image = load_picture(path_to_file)
image_shape = image.shape

faces = detect_faces(image)
faces_coordinates = converting_results_to_coordinates(faces, image_shape)
encoded_faces = converting_faces_to_array(image, faces_coordinates)

prediction = pred_mask(encoded_faces)
print(prediction)
draw_image(image, faces_coordinates, prediction)
