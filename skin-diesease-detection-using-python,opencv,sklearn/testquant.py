from os import listdir
from os.path import isfile, join
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
import cv2
import sklearn
from sklearn.metrics.pairwise import euclidean_distances 
import skimage.feature as sk
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import sys 
import matplotlib
from sklearn.metrics.cluster import entropy
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.vq import kmeans,vq
import scipy
import pandas as pd
from fpdf import FPDF
gog= []
oga=[]
mypath='./Acnedb'
ar=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
for im in ar:
	fet1=[]
	fet2=[]
	fet3=[]
	fetg=[]
	ac=[]

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

	pdf=FPDF()
	pdf.set_font("Arial")
	mypath='.\Acnedb'
	addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]

	ok=svm.SVC(C=0.5, kernel='rbf', degree=3, gamma=4, coef0=1.0, shrinking=False, probability=False, tol=0.01, cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
	kavi=svm.SVC(C=0.5, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=False, probability=False, tol=0.01, cache_size=300, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)

	mig=list(np.array(pd.read_csv('skin before.csv'))[:,2:])
	f16=list(np.array(pd.read_csv('skin after.csv'))[:,2:])
	f14=list(np.array(pd.read_csv('other param.csv'))[:,2:])

	acne=[]
	ork=[]
	for i in range(9):
		acne.append(list(f14[i]))
		acne.append(list(f14[9+i]))

	for i in range(36):
		ork.append(mig[i])
		ork.append(f16[i])
	o=np.array(ork)
	m=list(o)
	om=[]
	for i in range(36):
		om.append(0.0)
		om.append(1.0)
		
	kf=KFold(n_splits=35)
	for train, test in kf.split(m,om):

		
		xtt,xts=m[train[0]:train[-1]],m[test[0]:test[-1]]
		
		ytt,yts=om[train[0]:train[-1]],om[test[0]:test[-1]]
		
	ok.fit(m,om)
		
		
	elfg=[]
	elf=[]	

	for i in range(9):
		elf.append(0)
		elf.append(1)
	kavi.fit(acne,elf)

	for i in range(len(m)/2):
		elfg.append(ok.predict(m[2*i]))
		elfg.append(ok.predict(m[2*i+1]))
	luc=[]
	for i in range(9):
		luc.append(kavi.predict(acne[2*i]))
		luc.append(kavi.predict(acne[2*i+1]))
	for i in range(9):
		
		c=(abs(elfg[2*i]-elfg[2*i+1])+abs(elfg[2*i+18]-elfg[2*i+1+18])+abs(elfg[2*i+36]-elfg[2*i+1+36])+abs(elfg[2*i+54]-elfg[2*i+1+54])+abs(luc[2*i]-luc[2*i+1]))
	 	canberra=float(c)/(elfg[2*i]+elfg[2*i+1]+elfg[2*i+18]+elfg[2*i+1+18]+elfg[2*i+36]+elfg[2*i+1+36]+elfg[2*i+54]+elfg[2*i+1+54]+luc[2*i]+luc[2*i+1])
		p="improvement is "+str(list(canberra*100)[0])
		
	

	
	t=kavi.predict(ac)
	b=ok.predict(fet1[1:])
	c=ok.predict(fet2[1:])
	d=ok.predict(fet3[1:])
	e=ok.predict(fetg[1:])
	
	
	oga=[t,b,c,d,e]

	if t==1 and oga.count(1)>=2:
		lr=["clean image"]
	else:
		lr=["pimples are there"]
	gog.append(oga+lr)
		


for il in range(len(ar)):
	pdf.add_page()
	print ar[il],gog[il]
	pdf.image(ar[il])
	sri=gog[il][-1]	
	pdf.cell(0, 0,sri , 1, 1, 'C')
		
#	cv2.imshow(ar[il],cv2.imread(ar[il]))
#	cv2.waitKey(0)
pdf.output("Qfy.pdf")		 


		


	
	

	
