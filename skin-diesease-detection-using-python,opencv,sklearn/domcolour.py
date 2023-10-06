import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from os import listdir
from os.path import isfile, join
mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]



for d in addr:
	img=cv2.imread(d)
	img=cv2.resize(img,(128,128),)


	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h,s,v=cv2.split(img_hsv)

	lower_red = np.array([0,20,40])
	upper_red = np.array([40,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red = np.array([160,20,40])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

	# join my masks
	mask = mask0+mask1

	output_img = img.copy()
	output_img=cv2.bitwise_and(output_img,output_img,mask=mask)


	cv2.imshow("",output_img)
	cv2.waitKey(0)
	
	


	
