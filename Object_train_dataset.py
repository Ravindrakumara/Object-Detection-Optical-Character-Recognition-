import sys
import os
import cv2 as cv
import numpy as np


people = ['1','2','3','4','5','6','a','b','c','e','charlie chaplin','donald trump','joaquin phoenix','Nikola tesla']

DIR = r'C:\Users\B.Ravindra kumara\PycharmProjects\Computer_vision\train_data_set'

haar_cascade = cv.CascadeClassifier('haar_face_default.xml')

feature = []
lables = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3)
            for (x,y,w,h) in face_rect:
                faces_roi = gray [y:y+h, x:x+w]
                feature.append(faces_roi)
                lables.append(label)
                pass

create_train()
# print(f'number of face={len(feature)}')
# print(f'number of face={len(lables)}')
# print(f'DIR=',DIR)
print('Training Done ================!')
feature = np.array(feature,dtype='object')
lables = np.array(lables)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(feature,lables)

face_recognizer.save('Object_detect.yml')

np.save('feature.npy',feature)
np.save('feature.npy', lables)


    # pass
