import numpy as np
import cv2

from sklearn import tree
from sklearn.cross_validation import train_test_split

from os import listdir
from os.path import isfile, join

def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('D:\Sublime Text 3\Simple-skin-detection-master\Simple-skin-detection-master\data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv

def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    print trainData.shape
    print trainLabels.shape
    print testData.shape
    print testLabels.shape

    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(trainData, trainLabels)
    print clf.feature_importances_
    print clf.score(testData, testLabels)

    return clf

def ApplyToImage(path, flUseHSVColorspace):
    data, labels= ReadData()
    clf= TrainTree(data, labels, flUseHSVColorspace)

    img= cv2.imread(path)
    print img.shape
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))
    print data.shape

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))

    if (flUseHSVColorspace):
        cv2.imwrite('../results/result_HSV.png',((-(imgLabels-1)+1)*255))# from [1 2] to [0 255]
    else:
        cv2.imwrite('D:\Sublime Text 3\Simple-skin-detection-master\Simple-skin-detection-master/results/result_RGB.png',((-(imgLabels-1)+1)*255))





mypath='D:\Sublime Text 3\Acnedb'
addr=[mypath+"/"+fil for fil in listdir(mypath) if isfile(join(mypath, fil))]

for d in addr:
	ApplyToImage(d, True)
	ApplyToImage(d, False)

    