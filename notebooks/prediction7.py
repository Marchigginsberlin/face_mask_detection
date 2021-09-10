from facenet_pytorch import MTCNN
from PIL import Image
import torch
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
import cv2
import time
import numpy as np
import random

steps_threshold=[0.8, 0.8, 0.8]
detector = MTCNN(image_size = img_size, thresholds = steps_threshold)

#loading image in array
def load_image(filename, printing=False):
  # load image from file
  pixels = cv2.imread(filename) #format='jpeg')
  if printing == True:
    print("Shape of image/array:",pixels.shape)
    imgplot = plt.imshow(pixels)
    plt.show()
  return pixels

#detecting faces from image in array and getting the coordinates of it (with threshold)
def dectecting_faces(pixels):
  faces = detector.detect(pixels)
  return faces

#transforming coordinates of the face into a cropped image
def face_from_coordinates(img,x,y,w,h):
    return img[y:y+h,x:x+w]


#transforming all the detected coordinates in arrays and storing them into a dict
def converting_faces_to_array(pixels, faces, prob = False):
  encoded_faces = {}
  probabilities = []
  for index, face in enumerate(faces[0]):
      encoded_faces[f"face{index + 1}"] = face_from_coordinates(pixels, *face.astype(int))
      if prob == True:
        probabilities.append(faces[1][index])
        return encoded_faces, probabilities
  return encoded_faces


#resizing the output to 128 x 128
def resized_faces_array(encoded_faces):
  resized_faces_array = {}
  i = 1
  for keys, values in encoded_faces.items():
      individual_face =encoded_faces[f'face{i}']
      image=Image.fromarray(individual_face)
      image = image.resize((128,128))
      resized_faces_array[f'face{i}'] = np.asarray(image)
      i = i + 1
  return resized_faces_array


def output_mark_adriando(face):
  return random.randint(0, 3)


def predict_faces(faces):
  all_predictions = [output_mark_adriando(face) for face  in faces]
  return all_predictions


def draw_bounding_box(frame, faces_coordinates, predictions):
  color_corresp = {
      1: (0,0,255),
      2: (0,0,255),
      3: (0,0,255) 
  }

  for index, box in enumerate(faces_coordinates):
      cv2.rectangle(frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color = color_corresp[predictions[index]])


def run_video():
  cap = cv2.VideoCapture(0)
  #i = 0
  while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    #i = i + 1

    # Get each 4th frame
    #if i % 4 == 0:
      #pixels = load_image(frame)

    # Detect faces
    faces, probs = dectecting_faces(frame)
    if faces is not None:

    # Crop the faces as arrays
    encoded_faces = converting_faces_to_array(frame, encoded_faces)

    # Resize
    resized_faces = resized_faces_array(encoded_faces) # dict {'face1': ndarray,...}  

    # Get the prediction for each face (of whether it has a mask)
    predicted_faces = predict_faces(resized_faces)

    # Update the squares accordingly
    draw_bounding_box(frame, faces, predicted_faces)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
          break

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()

run_video()