from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time

cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    detector = MTCNN()
    result = detector.detect_faces(frame)
    if result q:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            #cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            #cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            #cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            #cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            #cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()