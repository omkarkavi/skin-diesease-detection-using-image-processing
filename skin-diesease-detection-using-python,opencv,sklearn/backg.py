import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import bisect







mypath='D:\Sublime Text 3\Acnedb/acneE1.jpg'
img=cv2.imread(mypath)
# find the keypoints with ORB
orb=cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img,None)
img = cv2.drawKeypoints(img, kp1, None, color=(0,255,0), flags=0)
cv2.imshow("jk",img)
cv2.waitKey(0)







