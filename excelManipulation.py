from cv2 import cv2
import os 
import numpy as np 
import facedetection as fr 
import csv
import pandas as pd
from time import sleep
from datetime import date

names={}
## get today to be insrted in excel sheet 
today = date.today()
# dd/mm/YY
d1 = today.strftime("%d/%m/%Y") 

def  fromExcelToCsv(filename):
    df = pd.read_excel(filename,index = False)
    df.to_csv('./data.csv')

def dataGetter():
    with open('data.csv','r') as f:
        data = csv.reader(f)
        next(data) # to get first rowwith names 
        rows = list(data)
        for row in rows:
            names[int(row[0])] =row[1]
    return names        

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

def updateExcel(filename):
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