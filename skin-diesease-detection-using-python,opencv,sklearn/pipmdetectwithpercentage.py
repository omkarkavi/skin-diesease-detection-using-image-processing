import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
acne=[]
for d in addr:
	img=cv2.imread(d)
	img=cv2.resize(img,(128,128),)
	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_red = np.array([0,10,50])
	upper_red = np.array([40,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	
	lower_red = np.array([160,10,50])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

	
	mask = mask0+mask1

	output_img = img.copy()
	output_img=cv2.bitwise_and(output_img,output_img,mask=mask)






	b,g,r=cv2.split(output_img)
	th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,7)
	im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	j=0
	for i in contours:
		
		if cv2.contourArea(i)>15 and cv2.contourArea(i)<150:
			j=j+1 
			cv2.drawContours(img,i,-1,(0,255,0),)
	acne.append(j)
	#cv2.imshow("",img)
	#cv2.waitKey(0)
mapp=list(zip(addr,acne))
for i in range(len(mapp)/2):
	print "For set "+str(i+1)+" percent improvement is "+str(float(mapp[2*i][1]-mapp[2*i+1][1])/mapp[2*i][1]*100.0)

