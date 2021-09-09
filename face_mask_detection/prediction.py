import tensorflow as tf
import cv2
import mtcnn
import os
from PIL import Image


def load_picture(path_to_file):
    img = tf.io.read_file(path_to_file)
    img = tf.image.decode_image(pixels, channels = 3, dtype = tf.float32)
    img = img.numpy()
    return img


detector = mtcnn.MTCNN(steps_threshold=[0.8, 0.8, 0.8])
def detect_faces(img):
    faces= detector.detect_faces(img*255)
    return faces


def face_from_coordinates(img,x,y,w,h):
    return img[y:y+h,x:x+w]

def face_cropping(faces, img):
    encoded_faces = []
    for index, face in enumerate(faces):
        encoded_faces.append(face_from_coordinates(img, *face['box']))
    return encoded_faces


def prep_prediction_multi(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    img= tf.expand_dims(img,0)
    return img

def prep_prediction_one(encoded_face):
    img = tf.image.resize_with_pad(encoded_face, 224, 224)
    #img= tf.expand_dims(img,0)
    return img

def pred_mask(faces, img):
    cropped_face= face_cropping(faces, img)
    if len(cropped_face) == 1:
        face = prep_prediction_one(cropped_face)
        return [model1.predict(face).argmax()]
    elif len(cropped_face) > 1:
        faces = []
        for j in range(len(cropped_face)):
            face = prep_prediction_multi(cropped_face[j])
            faces.append(face)
        predictions = []
        for face in faces:
            predictions.append(model1.predict(face).argmax())
        return predictions
    else:
        return [3]

def draw_image(image, coordinates, prediction):
    color_corresp = {
      0: (255,0,0),
      1: (255,165,0),
      2: (0,255,0)}
    coordinates =[faces[i]['box'] for i in range(len(faces))]
    for i in range(len(coordinates)):
        cv2.rectangle(image,
                      (coordinates[i][0],coordinates[i][1]),
                      (coordinates[i][0]+coordinates[i][2],coordinates[i][1]+ coordinates[i][3]),
                      color_corresp[prediction[i]],3)
    plt.imshow(image)
