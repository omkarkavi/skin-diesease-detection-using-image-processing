
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
from fpdf import FPDF
pdf=FPDF()
pdf.set_font("Arial")
mypath='.\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]

ok=svm.SVC()
kavi=svm.SVC()
mlp = MLPClassifier(verbose=0, random_state=0,max_iter=20,)
mig=list(np.array(pd.read_csv('skin before.csv'))[:,2:])
f16=list(np.array(pd.read_csv('skin after.csv'))[:,2:])
f14=list(np.array(pd.read_csv('other param.csv'))[:,2:])
acne=[]
ork=[]
for i in range(9):
	acne.append(f14[i])
	acne.append(f14[9+i])


for i in range(36):
	ork.append(mig[i])
	ork.append(f16[i])
o=np.array(ork)
m=list(o)
om=[]
for i in range(36):
	om.append(0.0)
	om.append(1.0)
	print o
kf=KFold(n_splits=2)
for train, test in kf.split(m,om):


	xtt,xts=m[36:],m[0:71]
	
	ytt,yts=om[36:],om[0:36]
	
	ok.fit(xtt,ytt)
	
	
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
	print p
	im1=cv2.imread(addr[2*i])
	im2=cv2.imread(addr[2*i+1])
	cv2.imshow("severe",im1)
	cv2.imshow("non severe",im2)
	cv2.waitKey(0)
	pdf.add_page()

	pdf.image(addr[2*i])
	pdf.image(addr[2*i+1])
	pdf.cell(0, 0, p, 1, 1, 'C')		
pdf.output("final.pdf","F")