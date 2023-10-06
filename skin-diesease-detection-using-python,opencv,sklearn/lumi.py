import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
for d in addr:
	img=cv2.imread(d)
	img=cv2.resize(img,(128,128),)
	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	lower_red = np.array([40,100,77])
	upper_red = np.array([255,175,127])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)


	# join my masks
	mask = mask0

	output_img = img.copy()
	output_img=cv2.bitwise_and(output_img,output_img,mask=mask)
	cv2.imshow("",output_img)
	cv2.waitKey(0)
