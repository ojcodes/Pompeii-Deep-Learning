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

#save_path= '/home/oj/oj_scripts/pictures/signs/resized/.jpg'
imgs = glob.glob("C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\*.jpg") #read all images
#np.random.shuffle(imgs)
print(len(imgs))

count=1 #counter for running x times or naming images
  
for file in imgs: #each file in this case in an image
    if count<=10: #temp loop to just run 10 images
        img=cv.imread(file,0) #create a variable that reads an image one by one (stores in the memory temporairly). 0 for BW. 1 for Color
        res_img = cv.resize(img, (700, 700))  # w,h
    
        #cv.imshow('img'+str(count),res_img)# shows img number
        #cv.imwrite('/home/oj/oj_scripts/pictures/signs/resized/s'+str(count)+'.jpg', res_img) #path+NewImageName,variable name//cv.write also needs cv.waitkey
        cv.imshow(str(file),res_img) #to display the image #. Helps in knowing whether the images have been shuffled
        cv.waitKey(3000) #in ms
        cv.destroyAllWindows()
        print (count)
        count = count+1
    