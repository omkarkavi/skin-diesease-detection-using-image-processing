from os import listdir
from os.path import isfile, join
import cv2
import sklearn
from sklearn.metrics.pairwise import euclidean_distances 
import skimage.feature as sk
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib
from sklearn.metrics.cluster import entropy
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import euclidean_distances
mig=[]
ecd=[]

conditions=["imge name and color","contrast","ASM","energy","homogeneity","dissimilarity","correlation","entropy"]
mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
for j in range(2):
	mig.append(j)
	if j==0:
		st="before"
	else:
		st="after"
	writer = pd.ExcelWriter('output with backgroud '+st+'.xlsx')
	fet1,fet2,fet3,fetg=[],[],[],[]
	for i in addr[j::2]:
		im1=cv2.imread(i)

		cv2.resize(im1,(256,256),interpolation = cv2.INTER_CUBIC)

		b,g,r =cv2.split(im1)
		nk1,nk2,nk3,nk=[],[],[],[]
		im2=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

		im3=cv2.cvtColor(im1,cv2.COLOR_BGR2HSV)
		#imp= cv2.adaptiveThreshold(im2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		glcm1 = sk.greycomatrix(b,[1],[0],symmetric=False,normed=True)
		glcm2 = sk.greycomatrix(r,[1],[0],symmetric=False,normed=True)
		glcm3 = sk.greycomatrix(g,[1],[0],symmetric=False,normed=True)
		glcmg = sk.greycomatrix(im2,[1],[0],symmetric=False,normed=True)
		glcm1=np.float32(glcm1)
		glcm2=np.float32(glcm2)
		glcm3=np.float32(glcm3)



		nk1.append(i+ " blue ")
		nk1.append(sk.greycoprops(glcm1,"contrast")[0][0])
		nk1.append(sk.greycoprops(glcm1,"ASM")[0][0])
		nk1.append(sk.greycoprops(glcm1,"energy")[0][0])
		nk1.append(sk.greycoprops(glcm1,"homogeneity")[0][0])
		nk1.append(sk.greycoprops(glcm1,"dissimilarity")[0][0])
		nk1.append(sk.greycoprops(glcm1,"correlation")[0][0])
		nk1.append(entropy(glcm1))
		fet1.append(nk1)
		


		nk2.append(i+" red ")	
		nk2.append(sk.greycoprops(glcm2,"contrast")[0][0])
		nk2.append(sk.greycoprops(glcm2,"ASM")[0][0])
		nk2.append(sk.greycoprops(glcm2,"energy")[0][0])
		nk2.append(sk.greycoprops(glcm2,"homogeneity")[0][0])
		nk2.append(sk.greycoprops(glcm2,"dissimilarity")[0][0])
		nk2.append(sk.greycoprops(glcm2,"correlation")[0][0])
		nk2.append(entropy(glcm2))
		fet2.append(nk2)
		
		nk3.append( i+" green ")	
		nk3.append(sk.greycoprops(glcm3,"contrast")[0][0])
		nk3.append(sk.greycoprops(glcm3,"ASM")[0][0])
		nk3.append(sk.greycoprops(glcm3,"energy")[0][0])
		nk3.append(sk.greycoprops(glcm3,"homogeneity")[0][0])
		nk3.append(sk.greycoprops(glcm3,"dissimilarity")[0][0])
		nk3.append(sk.greycoprops(glcm3,"correlation")[0][0])
		nk3.append(entropy(glcm3))
		fet3.append(nk3)


		nk.append(i+ " grey ")
		nk.append(sk.greycoprops(glcmg,"contrast")[0][0])
		nk.append(sk.greycoprops(glcmg,"ASM")[0][0])
		nk.append(sk.greycoprops(glcmg,"energy")[0][0])
		nk.append(sk.greycoprops(glcmg,"homogeneity")[0][0])
		nk.append(sk.greycoprops(glcmg,"dissimilarity")[0][0])
		nk.append(sk.greycoprops(glcmg,"correlation")[0][0])
		nk.append(entropy(glcmg))
		fetg.append(nk)


	non=fet1+fet2+fet3+fetg
	#non=np.array(non)
	#non=np.transpose(non)
		

		 
	df = pd.DataFrame(data = non, columns=conditions)

		
	df.to_excel(writer,"For skin sample "+str(j+1))

	stg="skin "+st+".csv"
	df.to_csv(stg)	
	
	mig[j]=pd.read_csv(stg)
	writer.save()


writer = pd.ExcelWriter('output distances with backgroud .xlsx')
before=np.array(mig[0])
after=np.array(mig[1])
gem=[]


for i in range(len(before)):
	
	k=np.array(before[i].tolist()[2:])
	g=np.array(after[i].tolist()[2:])

	gem.append((((g-k)**2)**0.5).tolist())
	

colour=[0,0,0,0]
colr=["blue","green","red","grey"]
adr=map(lambda x:"image"+x[-6],addr)


for i in range(len(colr)):	
	colour[i]=list(zip(conditions[1:],gem[6*i+0],gem[6*i+1],gem[6*i+2],gem[6*i+3],gem[6*i+4],gem[6*i+5]))

	dfb = pd.DataFrame(data = colour[i], columns=[""]+adr[::2])

	dfb.to_excel(writer,colr[i])
	
	writer.save()
		

			
			


	


	
	




 
