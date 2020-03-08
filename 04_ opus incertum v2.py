# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 20:37:48 2020
@author: osama
About: Work in progress version. Version 3 is a functional one.
"""
import glob
import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import cv2 as cv

#Unsorted images directiory
dir_path ='C:\\Users\\osama\\Desktop\\DATA 5000\\POMPEII PROJECT\\Python and Keras\\Project Images\\RAW Images\\raw\\Opus incertu _randomly shuffled by OJ into training_test_val'
#joining path and jpg
path = os.path.join(dir_path,'*jpg')
#loading images
images = glob.glob(path)
count=1
for i in images: #each file in this case in an image
    img=cv.imread(i,1) #create a variable that reads an image one by one (stores in the memory temporairly)
    res_img = cv.resize(img, (400, 400))  # w,h

    cv.imshow('img'+str(count),res_img)# shows img number
    #cv.imwrite('/home/oj/oj_scripts/pictures/signs/resized/s'+str(count)+'.jpg', res_img) #path+NewImageName,variable name//cv.write also needs cv.waitkey
    cv.waitKey(2)
    cv.destroyAllWindows()
    count = count+1

#create a folder "train": os.path.join gets path and the name of the new directory
#os.mkdir(directory_name_path) makes it
#You have to error check whether the folder already exist
#####Create a Test Dir#####
train_dir = os.path.join(dir_path, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)
####Create a Validation Dir#####
validation_dir = os.path.join(dir_path, 'validation') 
if os.path.exists(validation_dir):
    pass
else:
    os.mkdir(validation_dir)
####Create a Test Dir#####
test_dir = os.path.join(dir_path, 'test') 
if os.path.exists(test_dir):
    pass
else:
    os.mkdir(test_dir)
"""    
###BELOW DOES NOT WORK###
fnames = ['opus (' +str(i+1)+ ')' for i in range(200)]
for fname in fnames:
    src = os.path.join(dir_path,fname)
    dst= os.path.join (train_dir,fname)
    shutil.copyfile(src,dst)
"""