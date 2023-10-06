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
from scipy.cluster.vq import kmeans,vq
import scipy
import pandas as pd
from fpdf import FPDF 

o=input("Enter sample number between 1 to 9 ")

conditions=["imge name and color","contrast","ASM","energy","homogeneity","dissimilarity","correlation","entropy"]
mypath='.\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]
im1=cv2.imread(addr[i])
cv2.imshow("",im1)
im2=cv2.imread(addr[2*i-1])