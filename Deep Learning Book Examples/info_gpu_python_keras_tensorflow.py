# -*- coding: utf-8 -*-
"""
Spyder Editor
Tells info about gpu,cuda, cudnn, keras, python and tensor flow version
This is a temporary script file.
"""

# SETTING UP TENSORFLOW AND KERAS
#tf, python and CUDA compatibility variants
#https://www.tensorflow.org/install/source#linux 
#https://www.tensorflow.org/install/source_windows#tested_build_configurations

import sys;print('python version:',sys.version)
import tensorflow as tf; print('tf_version:',tf.__version__) #tf: 2.1
from tensorflow import keras; print('keras version:',keras.__version__)
from tensorflow.python.platform import build_info as tf_build_info
# print('CUDA Version:',tf_build_info.cuda_version_number)
# print('CUDNN Version:',tf_build_info.cudnn_version_number)




#Releases the python gpu memory
from numba import cuda
cuda.select_device(0)
cuda.close()

#cat /proc/meminfo ##Checks the RAM memory

import tensorflow as tf
tf.test.is_gpu_available() # True/False

# Or only check for gpu's with cuda support
tf.test.is_gpu_available(cuda_only=True) 
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
