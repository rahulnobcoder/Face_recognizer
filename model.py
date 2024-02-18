import cv2 as cv
import os 
import numpy as np 
haar_cascade=cv.CascadeClassifier('haar_face.xml')
features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy',allow_pickle=True)
def train(dir,p):
    haar_cascade=cv.CascadeClassifier('haar_face.xml')
    features=[]
    labels=[]
    for person in p:
        path=os.path.join(dir,person)
        label=p.index(person)
        for image in os.listdir(path):
            img_path=os.path.join(path,image)

            img_array=cv.imread(img_path,0)

            faces_rect=haar_cascade.detectMultiScale(img_array,scaleFactor=1.2,minNeighbors=6)
            for (x,y,w,h) in faces_rect:
                faces_roi = img_array[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
    
    features=np.array(features,dtype='object')
    labels=np.array(labels)
    print(len(features))
    print(len(labels))
    face_recognizer=cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(features,labels)

    face_recognizer.save('face_train.yml')
    np.save('features.npy',features)
    np.save('labels.npy',labels)
    return face_recognizer,features,labels

def predict(img,c_img,p):
    haar_cascade=cv.CascadeClassifier('haar_face.xml')
    features=np.load('features.npy',allow_pickle=True)
    labels=np.load('labels.npy',allow_pickle=True)
    face_rec=cv.face.LBPHFaceRecognizer_create()
    face_rec.read('face_train.yml')
    faces_rect=haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    if len(faces_rect)==0:
        return c_img
    for (x,y,w,h) in faces_rect:
        faces_roi=img[y:y+h,x:x+w]
        label,confidence=face_rec.predict(faces_roi)
        print(label)
        cv.putText(c_img,str(p[label]),(x-10,y-10),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
        cv.rectangle(c_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        return c_img
    
    


