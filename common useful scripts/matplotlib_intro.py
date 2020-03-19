#Testing the basics of matplotlib. It is like matlab plots. We will use .pyplot and .image

# import .pyplot for the mathematical plots
import  matplotlib.pyplot as plt 
x=[1,2,3,4,5,6]
y=[2,4,8,16,32,64]
plt.plot(x,y)
plt.ylabel('y-axis..')
plt.show()

# import .image for the mathematical plots
import matplotlib.image as mpimg
#Give the right image path. Seems like \\ is required for the path in python
img_path = 'C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\test\\test_shuf_193.JPG'
a= mpimg.imread(img_path)
plt.imshow(a)
