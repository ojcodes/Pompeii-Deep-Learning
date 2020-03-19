#Testing the basics of matplotlib. It is like matlab plots. We will use .pyplot and .image


# # import .image for the mathematical plots
# import matplotlib.image as mpimg
# #Give the right image path. Seems like \\ is required for the path in python
# img_path = 'C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\test\\test_shuf_193.JPG'
# a= mpimg.imread(img_path)
# plt.imshow(a)


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time

#matplotlib inline

PATH = 'C:\\Users\\osama\\Desktop\\del\\{}.jpg' #has to have 1.jpg, 2.jpg etc

for i in range(1,4):
    p = PATH.format(i)
    #print p
    image = mpimg.imread(p) # images are color images
    plt.gca().clear()
    plt.imshow(image);
    display.display(plt.gcf())
    #display.clear_output(wait=True)
    time.sleep(0.5) # wait one second
    