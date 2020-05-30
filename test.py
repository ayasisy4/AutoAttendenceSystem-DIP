from cv2 import cv2
import os 
import numpy as np 
import facedetection as fr 

testImage = cv2.imread('/Users/Aya/AutoAttendenceSystem-DIP/TestImages/alluni.jpg')
detectedFaces, grayImage = fr.facedetetection(testImage)
print (detectedFaces)

# for(x,y,w,h ) in detectedFaces:
#     cv2.rectangle(testImage,(x,y),(x+w,y+h),(255,0,0),thickness=5)

# resized_img= cv2.resize(testImage,(1000,700))

# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("face detection tutorial", resized_img)
# cv2.waitKey()

##to train a model 

faces,faceID = fr.labels_for_training_data("/Users/Aya/AutoAttendenceSystem-DIP/trainingImages")
facerecongizer = fr.train_classifier(faces,faceID) 
facerecongizer.save('trainingData.yml')

#to test alread trained model 

# facerecongizer= cv2.face.LBPHFaceRecognizer_create()
# facerecongizer.read('/Users/Aya/AutoAttendenceSystem-DIP/trainingData.yml')
name={0: "rawda",
1: "yasmeen",
2: "sohaila",
3:"aya"}

for face in detectedFaces:
    (x,y,w,h)= face
    roi_gray= grayImage[y:y+h,x:x+h]
    label,confidence= facerecongizer.predict(roi_gray)
    print("confidnece: ", confidence)
    print("label: ", label)
    fr.draw_rect(testImage,face)
    predictedName= name[label]
    if(confidence < 90 ):
        continue
    fr.put_text(testImage,predictedName,x,y)

resized_img= cv2.resize(testImage,(1200,1200))

cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("face detection result", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()