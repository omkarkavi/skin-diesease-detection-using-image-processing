import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize






mypath='D:\Sublime Text 3\Acnedb/acneE1.jpg'
im=cv2.imread(mypath)

img=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
h,s,v=im[:,:,0],im[:,:,1],im[:,:,2]




_,th2 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
ad=img.tolist()

for eve in ad:
	for i in eve:
		for j in i:
			if j<np.mean(img):
				j=0




cv2.imshow('rt',th2)
cv2.waitKey(0)


