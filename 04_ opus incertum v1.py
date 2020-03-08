# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:56:17 2020
unuseful
@author: osama

Game Plan:
1) Start with the normal opus_incertum images. Clean the data.
2) Try with Jeff's created images and see what you get'

Spyder IDE Commands:
 1) clc clears the console
"""
import sys
print (sys.version)

#Following page 132 to distribute images into training, validation and test data
import os, shutil

#sets the path - double \\ needed for some reason
base_dir ='C:\\Users\\osama\\Desktop\\DATA 5000\\POMPEII PROJECT\\Python and Keras\Project Images\\RAW Images\\raw\\opus_incertum_original' 

#create a folder "train": os.path.join gets path and the name of the new directory
#os.mkdir(directory_name_path) makes it
#You have to error check whether the folder already exist

#####Create a Test Dir#####
train_dir = os.path.join(base_dir, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)

#####Create a Validation Dir#####
validation_dir = os.path.join(base_dir, 'validation') 
if os.path.exists(validation_dir):
    pass
else:
    os.mkdir(validation_dir)

####Create a Test Dir#####
test_dir = os.path.join(base_dir, 'test') 
if os.path.exists(test_dir):
    pass
else:
    os.mkdir(test_dir)