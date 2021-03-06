# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:05:36 2020

@author: osama
"""

# This script reads images (folder) from a destination, resizes it and saves at a destination
import time
import cv2 as cv
import glob
import numpy as np
import os, shutil
# import tensorflow as tf ##to check if things are configured 
# print (tf.__version__)
#save_path= '/home/oj/oj_scripts/pictures/signs/resized/.jpg'

dir_path="C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled" #has all unsorted images

#create a folder "train": os.path.join gets path and the name of the new directory
#os.mkdir(directory_name_path) makes it
#You have to error check whether the folder already exist

#####Create a Test Dir#####
train_dir = os.path.join(dir_path, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)
#####Create a Validation Dir#####
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

#GET IMAGES AND SHUFFLE
path = os.path.join(dir_path,'*jpg') #Adding *.jpg to the path
imgs = glob.glob(path);print('images in base_dir=',len(imgs))#read all images
np.random.shuffle(imgs) # to randomize images as per pg#132 Deep Learning F. Chollet. Allows to have similar rep. in train and test set

#DISPLAY AND SAVE IMAGES
count=1 #counter for running x times or naming images
for file in imgs: #each file in this case in an image
    if count<=4: #temp loop to just run 10 images
        img=cv.imread(file,-1) #create a variable that reads an image one by one (stores in the memory temporairly). 0 for BW. 1 for Color
        res_img = cv.resize(img, (700, 700))  # w,h
    
        #cv.imshow('img'+str(count),res_img)# shows img number
        
        cv.imshow(str(file),res_img) #to display the image #. Helps in knowing whether the images have been shuffled
        #cv.imwrite(train_dir, res_img)
        cv.waitKey(10) #in ms
        cv.destroyAllWindows()
        print (count)
        count = count+1

"""
#trying to put images into train, val and test folders 160, 20 and 20 resp.
fname = ['*.jpg'.format(i) for i in range(160)] #creartes a list. fname is a var_name
for i in fname:
    src = os.path.join(basedir_path,fname)
    dst= os.path.join (train_dir,fname)
    shutil.copyfile(src,dst)
"""
