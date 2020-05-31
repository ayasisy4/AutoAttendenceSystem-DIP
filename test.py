from cv2 import cv2
import os 
import numpy as np 
import facedetection as fr 
import csv
import pandas as pd
from time import sleep
from datetime import date

today = date.today()

# dd/mm/YY
d1 = today.strftime("%d/%m/%Y")

testImage = cv2.imread('/Users/Aya/AutoAttendenceSystem-DIP/TestImages/aya.jpg')
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
namemanual={0: "rawda",
1: "yasmeen",
2: "sohaila",
3:"aya"}

filename = 'E:/backendData.xlsx'

names = {}
labels = []
students = []
finalrows=[]
# function to deal with excel 
def  fromExcelToCsv():
    df = pd.read_excel(filename,index = False)
    df.to_csv('./data.csv')

def dataGetter():
    with open('data.csv','r') as f:
        data = csv.reader(f)
        next(data) # to get first rowwith names 
        rows = list(data)
        for row in rows:
            names[int(row[0])] =row[1]

def attended(recognizedstudent):
    with open('data.csv','r') as f :
        data = csv.reader(f)
        # row0 = next(data)
        # row0.append(d1)
        # print(row0)
        rows = list(data)    
        print('kkkkkkkkkk',rows[0])
        for row in rows:
            if(row[1]=='Name'):
                row.append(d1)
            else:
                row.append('0')
        print('this is all data before any modi',row)

        for row in rows:
            print("row name ,", row[1])
            if(row[1] in recognizedstudent ):
                row[-1]='1'
    print("row after appednig attendce ,", rows)

    with open('data.csv','w') as g:
            writer = csv.writer(g,lineterminator='\n')
            writer.writerows(rows)
# def writeincsv (latestdata): 
#         with open('data.csv','w') as g:
#             writer = csv.writer(g,lineterminator='\n')
#             writer.writerows(latestdata)
                    
def updateExcel():
        with open('data.csv') as f:
            data = csv.reader(f)
            lines = list(data)
            for line in lines:
                line.pop(0)
            with open('data.csv','w') as g:
                writer = csv.writer(g,lineterminator='\n')
                writer.writerows(lines)
                
        df = pd.read_csv('data.csv')
        df.to_excel(filename,index = False)


fromExcelToCsv() # converting the excel to csv for use
dataGetter() # getting the data from csv in a dictionary
print('Total students :',names)

for face in detectedFaces:
    (x,y,w,h)= face
    roi_gray= grayImage[y:y+h,x:x+h]
    label,confidence= facerecongizer.predict(roi_gray)
    print("confidnece: ", confidence)
    print("label: ", label)
    fr.draw_rect(testImage,face)
    predictedName= namemanual[label]
    if(confidence < 37 ):
        continue
    fr.put_text(testImage,predictedName,x,y)
    labels.append(label)
    students.append(namemanual[label])
    totalstudents = set(students)
    justlabels = set(labels)
    print('student Recognised : ',students,totalstudents,justlabels)
for i in justlabels:
    print('label count : ', labels.count(i))
    print("names of i is ", names[i])
    
attended(totalstudents)


# writeincsv(finalrows)
resized_img= cv2.resize(testImage,(1200,1200))
updateExcel()
cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("face detection result", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()