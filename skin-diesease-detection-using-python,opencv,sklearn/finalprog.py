from os import listdir
from os.path import isfile, join
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

mig=[]
ecd=[]
acne=[]
mean=[]
var=[]
pdf=FPDF()
pdf.set_font("Arial")
conditions=["imge name and color","contrast","ASM","energy","homogeneity","dissimilarity","correlation","entropy"]
mypath='.\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]

m=1
for j in range(2):
	mig.append(j)
	if j==0:
		st="before"
	else:
		st="after"
	writer = pd.ExcelWriter('output without background '+st+'.xlsx')
	fet1,fet2,fet3,fetg=[],[],[],[]
	for i in addr[j::2]:
		

		im1=cv2.imread(i)
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
		cv2.imwrite(str(m)+".jpg",output_img)
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
				
		cv2.imwrite(str(100+m)+".jpg",im1)
		
		acne.append(no_coun)
		
		
		img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		h,s,v=cv2.split(img_hsv)

		
		
		mom2=(np.sum((h-np.mean(h))**2)/np.size(h))**0.5
		mom3=np.mean(h)
		var.append(mom2)
				
		mean.append(mom3)


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
		m=m+1
 
		

	non=fet1+fet2+fet3+fetg

		

		 
	df = pd.DataFrame(data = non, columns=conditions)

		
	df.to_excel(writer,"For skin samples "+st)

	stg="skin "+st+".csv"
	df.to_csv(stg)	
	
	mig[j]=pd.read_csv(stg)
	writer.save()


writer = pd.ExcelWriter('output distances after background removal.xlsx')
before=np.array(mig[0])
after=np.array(mig[1])
gem=[]
kem=[]
sem=[]
for i in range(len(before)):
	
	k=np.array(before[i].tolist()[2:])
	g=np.array(after[i].tolist()[2:])
	kem.append(abs(g-k)/(g+k)*100)
	gem.append(((((g-k)**2)**0.5)/k*100).tolist())
	sem.append(abs(g-k))

colour=[0,0,0,0]
colr=["blue","green","red","grey"]
adr=map(lambda x:"Sample "+chr(ord('a')+x),range(len(addr)/2))
canbera=["bluecanberra","greencanberra","redcanberra","greycanberra"]
manhattan=["blueman","greenman","redman","greyman"]	


for i in range(len(colr)):
	
	colour[i]=gem[len(adr)*i:len(adr)*i+len(adr)]
	
	
	dfb = pd.DataFrame(data = colour[i], columns=conditions[1:])

	dfb.to_excel(writer,colr[i])
	colour[i]=kem[len(adr)*i:len(adr)*i+len(adr)]
	
	
	dfb = pd.DataFrame(data = colour[i], columns=conditions[1:])

	dfb.to_excel(writer,canbera[i])
	colour[i]=kem[len(adr)*i:len(adr)*i+len(adr)]
	
	
	dfb = pd.DataFrame(data = colour[i], columns=conditions[1:])

	dfb.to_excel(writer,canbera[i])	
	colour[i]=sem[len(adr)*i:len(adr)*i+len(adr)]
	
	
	dfb = pd.DataFrame(data = colour[i], columns=conditions[1:])

	dfb.to_excel(writer,manhattan[i])

cu1=[]
cu2=[]
for i in range(1,len(addr)/2+1):
	pdf.add_page()
	pdf.set_title("Acnedb")
	pdf.image(addr[2*i-2])
	pdf.image(str(i)+".jpg")
	pdf.image(str(100+i)+".jpg")
	pdf.add_page()
	pdf.image(addr[2*i-1])
	pdf.image(str(i+len(addr)/2)+".jpg")
	pdf.image(str(100+i+len(addr)/2)+".jpg")
	cu1.append("Sample "+chr(ord('a')+i-1)+" before")
	cu2.append("Sample "+chr(ord('a')+i-1)+" after")
	cu=cu1+cu2
imp1=[]
imp2=[]
imp3=[]
imp4=[]
for i in range(len(addr)/2):
	imp1.append(abs(float(acne[i]-acne[i+len(addr)/2]))/acne[i]*100)
	
	imp2.append(abs(float(var[i]-var[i+len(addr)/2]))/var[i]*100.0)
	imp3.append(abs(float(mean[i]-mean[i+len(addr)/2]))/mean[i]*100.0)
	
imp=list(zip(adr,imp1,imp2,imp3))
dfb = pd.DataFrame(data =imp , columns=["sample","improvement in acne","improvement in 2nd moment","improvement in 1st moment"])
dfb.to_excel(writer," improvement in other params")


writer.save()
print acne
writer = pd.ExcelWriter('Acne,mean and variance.xlsx')
df=pd.DataFrame(data=list(zip(cu,acne,mean,var)),columns=["sample","no of pimples","mean","varience"])
df.to_excel(writer,"other params")
df.to_csv("other param.csv")
writer.save()

pdf.output("Acne.pdf","F")