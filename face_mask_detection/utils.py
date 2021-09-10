from facenet_pytorch import MTCNN
from PIL import Image
import torch
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
import cv2
import time
import numpy as np
import random


#detecting faces from image in array and getting the coordinates of it (with threshold)
def dectecting_faces(pixels):
  faces = detector.detect(pixels)
  return faces

#transforming coordinates of the face into a cropped image
def face_from_coordinates(img,x1,y1,x2,y2):
    print((x1, y1), (x2, y2))
    return img[max(y1, 0):y2,max(x1, 0):x2]


#transforming all the detected coordinates in arrays and storing them into a dict
def converting_faces_to_array(pixels, faces_boxes, prob = False):
  encoded_faces = {}
  probabilities = []
  for index, face in enumerate(faces_boxes):
      encoded_faces[f"face{index + 1}"] = face_from_coordinates(pixels, *face.astype(int))
      if prob == True:
        #probabilities.append(faces[1][index])
        return encoded_faces#, probabilities
  return encoded_faces


#resizing the output to 128 x 128
def resized_faces_array(encoded_faces):
  resized_faces_array = {}
  for key, individual_face in encoded_faces.items():
      image = Image.fromarray(individual_face)
      image = image.resize((128,128))
      resized_faces_array[key] = np.asarray(image)
  return resized_faces_array


def output_mark_adriando(face):
  return random.randint(1, 3)


def predict_faces(faces):
  all_predictions = [output_mark_adriando(face) for face  in faces]
  return all_predictions


def draw_bounding_box(frame, faces_coordinates, predictions):
  color_corresp = {
      1: (0,0,255),
      2: (0,255,0),
      3: (255,0,0)
  }

  for index, box in enumerate(faces_coordinates):
      cv2.rectangle(frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color = color_corresp[predictions[index]])
