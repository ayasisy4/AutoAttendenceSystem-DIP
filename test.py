from cv2 import cv2
import os 
import numpy as np 
import facedetection as fr 
import csv
import pandas as pd
from time import sleep
from datetime import date
import excelManipulation as excel
import imageEnhancing as imgENHC

# get the image that We  here consider as an attendnce photo
testImage = cv2.imread('/Users/Aya/AutoAttendenceSystem-DIP/TestImages/alex.jpg')
preImage4 = imgENHC.HistogramEqualization(testImage)

# 

#detect face in this image 
detectedFaces, grayImage = fr.facedetetection(preImage4)
print (detectedFaces)

# for(x,y,w,h ) in detectedFaces:
#     cv2.rectangle(testImage,(x,y),(x+w,y+h),(255,0,0),thickness=5)

# resized_img= cv2.resize(testImage,(1000,700))

# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("face detection tutorial", resized_img)
# cv2.waitKey()

##save the trained model 

faces,faceID = fr.labels_for_training_data("/Users/Aya/AutoAttendenceSystem-DIP/trainingImages")
facerecongizer = fr.train_classifier(faces,faceID) 
facerecongizer.save('trainingData.yml')

#to test alread trained model 

# facerecongizer= cv2.face.LBPHFaceRecognizer_create()
# facerecongizer.read('/Users/Aya/AutoAttendenceSystem-DIP/trainingData.yml')


#was for trial 
# namemanual={0: "rawda",
# 1: "yasmeen",
# 2: "sohaila",
# 3:"aya"}
#backendData
filename = 'E:/backendData.xlsx'
#variables
names = {}
labels = []
students = []
finalrows=[]
# functions to deal with excel 


#code begin 
excel.fromExcelToCsv(filename) # converting the excel to csv for use
names= excel.dataGetter() # getting the data from csv in a dictionary and fill in variables
print('Total students :',names)
def recogize_label_faces():

    for face in detectedFaces:
        (x,y,w,h)= face
        roi_gray= grayImage[y:y+h,x:x+h]
        label,confidence= facerecongizer.predict(roi_gray)
        print("confidnece: ", confidence)
        print("label: ", label)
        fr.draw_rect(preImage4,face)
        predictedName= names[label]
        if(confidence < 37 ):
            continue
        fr.put_text(preImage4,predictedName,x,y)
        labels.append(label)
        students.append(names[label])
        totalstudents = set(students)
        justlabels = set(labels)
        print('student Recognised : ',students,totalstudents,justlabels)
    for i in justlabels:
        print('label count : ', labels.count(i))
        print("names of i is ", names[i])
    
    excel.attended(totalstudents)
    excel.updateExcel(filename)
    
recogize_label_faces() #recognition function

##preprocessing Image 
preImage= imgENHC.grayingImage(testImage)
preImage2= imgENHC.adjust_gamma(preImage)
preImage3= imgENHC.DoG(testImage)
resized_img= cv2.resize(testImage,(1200,1200))
cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("face detection result", preImage4)
cv2.waitKey()
cv2.destroyAllWindows()