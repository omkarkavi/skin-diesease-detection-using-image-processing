from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import cv2
import sklearn
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import sys 
import matplotlib
from sklearn.metrics.cluster import entropy
import skimage.feature as sk
mig=list(np.array(pd.read_csv('skin before.csv'))[:,2:])
f16=list(np.array(pd.read_csv('skin after.csv'))[:,2:])
x25=mig+f16
f14=list(np.array(pd.read_csv('other param.csv'))[:,2:])
f14=np.array(f14)
pim=f14[:,0].reshape(18,1)
print pim
p=np.array([[42],[53],[120]])
feture=np.array([[148.6101491877,0.0386320576,0.1346023096,0.3810989295,4.4902684343,0.965603337,0.479500503],[99.0637666912,0.0391451508,0.1512169472,0.4793333962,3.1111968113,0.9796562144,0.412019357],[188.044463023,0.0441343225,0.1742386028,0.5051485972,3.7467131609,0.9403796436,0.4039250432]])
m=KMeans(n_clusters=3, init=feture, n_init=10, max_iter=50, tol=0.01, precompute_distances='auto', verbose=1, random_state=None, copy_x=True, n_jobs=1, algorithm='auto').fit(x25)
om=KMeans(n_clusters=3, init=p, n_init=10, max_iter=50, tol=0.01, precompute_distances='auto', verbose=1, random_state=None, copy_x=True, n_jobs=1, algorithm='auto').fit(pim)
omk=KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=50, tol=0.01, precompute_distances='auto', verbose=1, random_state=None, copy_x=True, n_jobs=1, algorithm='auto').fit(f14[:,1:])  
mypath='./testacne'
ar=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
elf=[]
for im in ar:
	fet1=[]
	fet2=[]
	fet3=[]
	fetg=[]
	ac=[]
	ilem=[]
	im1=cv2.imread(im)
	im1=cv2.resize(im1,(250,250),)
			
			
	img_hsv=cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)


	lower_red = np.array([0,10,100])
	upper_red = np.array([40,255,255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)


	lower_red = np.array([140,10,100])
	upper_red = np.array([180,255,255])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

			
	mask = mask0+mask1

			
	output_img = im1.copy()
	output_img=cv2.bitwise_and(output_img,output_img,mask=mask)

	img=output_img.copy()
			
			

	b,g,r =cv2.split(output_img)
			
	th = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,6)

	im2, contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
			
	no_coun=0
			
			
	for h in contours:
				
				
		x,y,w,hie=cv2.boundingRect(h)
				
				 
		if cv2.contourArea(h)>20 and cv2.contourArea(h)<220 :
				
			no_coun=no_coun+1 
			cv2.drawContours(im1,h,-1,(0,255,0),)
					

			
	ac.append(no_coun)
			
			
	img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h,s,v=cv2.split(img_hsv)

			
			
	mom2=(np.sum((h-np.mean(h))**2)/np.size(h))**0.5
	mom3=np.mean(h)
	ac.append(mom2)
					
	ac.append(mom3)


	nk1,nk2,nk3,nk=[],[],[],[]
	im2=cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)

	b,g,r=cv2.split(output_img)

			
	glcm1 = sk.greycomatrix(b,[1],[0],symmetric=False,normed=True)
	glcm2 = sk.greycomatrix(r,[1],[0],symmetric=False,normed=True)
	glcm3 = sk.greycomatrix(g,[1],[0],symmetric=False,normed=True)
	glcmg = sk.greycomatrix(im2,[1],[0],symmetric=False,normed=True)
	glcm1=np.float32(glcm1)
	glcm2=np.float32(glcm2)
	glcm3=np.float32(glcm3)



	nk1.append(im+ " blue ")
	nk1.append(sk.greycoprops(glcm1,"contrast")[0][0])
	nk1.append(sk.greycoprops(glcm1,"ASM")[0][0])
	nk1.append(sk.greycoprops(glcm1,"energy")[0][0])
	nk1.append(sk.greycoprops(glcm1,"homogeneity")[0][0])
	nk1.append(sk.greycoprops(glcm1,"dissimilarity")[0][0])
	nk1.append(sk.greycoprops(glcm1,"correlation")[0][0])
	nk1.append(entropy(glcm1))
	fet1=nk1
			


	nk2.append(im+" red ")	
	nk2.append(sk.greycoprops(glcm2,"contrast")[0][0])
	nk2.append(sk.greycoprops(glcm2,"ASM")[0][0])
	nk2.append(sk.greycoprops(glcm2,"energy")[0][0])
	nk2.append(sk.greycoprops(glcm2,"homogeneity")[0][0])
	nk2.append(sk.greycoprops(glcm2,"dissimilarity")[0][0])
	nk2.append(sk.greycoprops(glcm2,"correlation")[0][0])
	nk2.append(entropy(glcm2))
	fet2=nk2
			
	nk3.append( im+" green ")	
	nk3.append(sk.greycoprops(glcm3,"contrast")[0][0])
	nk3.append(sk.greycoprops(glcm3,"ASM")[0][0])
	nk3.append(sk.greycoprops(glcm3,"energy")[0][0])
	nk3.append(sk.greycoprops(glcm3,"homogeneity")[0][0])
	nk3.append(sk.greycoprops(glcm3,"dissimilarity")[0][0])
	nk3.append(sk.greycoprops(glcm3,"correlation")[0][0])
	nk3.append(entropy(glcm3))
	fet3=nk3


	nk.append(im+ " grey ")
	nk.append(sk.greycoprops(glcmg,"contrast")[0][0])
	nk.append(sk.greycoprops(glcmg,"ASM")[0][0])
	nk.append(sk.greycoprops(glcmg,"energy")[0][0])
	nk.append(sk.greycoprops(glcmg,"homogeneity")[0][0])
	nk.append(sk.greycoprops(glcmg,"dissimilarity")[0][0])
	nk.append(sk.greycoprops(glcmg,"correlation")[0][0])
	nk.append(entropy(glcmg))
	fetg=nk
	ilem.append(m.predict(fet1[1:]))
	ilem.append(m.predict(fet2[1:]))
	ilem.append(m.predict(fet3[1:]))
	ilem.append(m.predict(fetg[1:]))
	ilem.append(om.predict(ac[0]))
	
	elf.append(ilem)

for i in range(len(elf)):
	clas=0

	
	for j in range(len(elf[i])):
		if j==len(elf[i])-1:
		 	clas=clas+32*elf[i][j]
		else:
			clas=clas+2*elf[i][j]
	print clas
	if clas>28:
		print 'severe'
	elif clas<=28 and clas>=8:
		print 'medium'
	else:
		print 'normal'
	cv2.imshow("",cv2.imread(ar[i])) 
	cv2.waitKey(0)
