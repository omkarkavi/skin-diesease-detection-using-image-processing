import cv2
import sklearn
import skimage.feature as sk
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib
mig=[]
colr=[1,2,3,4]
writer = pd.ExcelWriter('output.xlsx')
conditions=["contrast","ASM","energy","homogeneity","dissimilarity","correlation"]
addr=['D://Ringworm//a123.jpg','D://Ringworm//a456.jpg','D://Ringworm//a789.jpg','D://Ringworm//a901.jpg']
for i in addr:
	im1=cv2.imread(i)

	cv2.resize(im1, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

	b,g,r =cv2.split(im1)
	nk1,nk2,nk3,fet1,fet2,fet3=[],[],[],[],[],[]
	im2=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

	im3=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)
	#imp= cv2.adaptiveThreshold(im2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	glcm1 = sk.greycomatrix(b,[1],[0],symmetric=False,normed=True)
	glcm2 = sk.greycomatrix(r,[1],[0],symmetric=False,normed=True)
	glcm3 = sk.greycomatrix(g,[1],[0],symmetric=False,normed=True)
	glcm1=np.float32(glcm1)
	glcm2=np.float32(glcm2)
	glcm3=np.float32(glcm3)



	
	nk1.append(sk.greycoprops(glcm1,"contrast")[0][0])
	nk1.append(sk.greycoprops(glcm1,"ASM")[0][0])
	nk1.append(sk.greycoprops(glcm1,"energy")[0][0])
	nk1.append(sk.greycoprops(glcm1,"homogeneity")[0][0])
	nk1.append(sk.greycoprops(glcm1,"dissimilarity")[0][0])
	nk1.append(sk.greycoprops(glcm1,"correlation")[0][0])
	fet1.append(nk1)
	print nk1


	
	nk2.append(sk.greycoprops(glcm2,"contrast")[0][0])
	nk2.append(sk.greycoprops(glcm2,"ASM")[0][0])
	nk2.append(sk.greycoprops(glcm2,"energy")[0][0])
	nk2.append(sk.greycoprops(glcm2,"homogeneity")[0][0])
	nk2.append(sk.greycoprops(glcm2,"dissimilarity")[0][0])
	nk2.append(sk.greycoprops(glcm2,"correlation")[0][0])
	fet2.append(nk2)
	print nk2
	
	nk3.append(sk.greycoprops(glcm3,"contrast")[0][0])
	nk3.append(sk.greycoprops(glcm3,"ASM")[0][0])
	nk3.append(sk.greycoprops(glcm3,"energy")[0][0])
	nk3.append(sk.greycoprops(glcm3,"homogeneity")[0][0])
	nk3.append(sk.greycoprops(glcm3,"dissimilarity")[0][0])
	nk3.append(sk.greycoprops(glcm3,"correlation")[0][0])
	fet3.append(nk3)
    print nk3 


	
	
#	df = pd.DataFrame(data = non, columns=["blue1","red1","green1"])
	
#	df.to_excel(writer,"For skin sample "+str(colr[addr.index(i)]))
	
#writer.save()



 
