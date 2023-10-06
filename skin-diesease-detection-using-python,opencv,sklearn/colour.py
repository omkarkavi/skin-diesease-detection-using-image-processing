import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from os import listdir
from os.path import isfile, join
mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
n=[]
for d in addr:
	
	img=cv2.imread(d)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	h,s,v=cv2.split(img)
	mom1=np.mean(h)
	mom2=np.var(h)
#	print mom1,mom2
	n.append([mom1,mom2])
print n
j=[]
for i in range(len(n)/2):
	j.append((n[i*2][1]-n[i*2+1][1])/n[i*2][1]*100)
print j


