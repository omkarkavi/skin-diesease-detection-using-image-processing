from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
ok=svm.SVC()
mlp = MLPClassifier(verbose=0, random_state=0,max_iter=20,)
mig=list(np.array(pd.read_csv('skin before.csv'))[:,2:])
f16=list(np.array(pd.read_csv('skin after.csv'))[:,2:])
f14=list(np.array(pd.read_csv('other param.csv'))[:,2:])
print f14
ork=[]
acne=[]
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
	om.append(0)
	om.append(1)
kf=KFold(n_splits=2)
for train, test in kf.split(m,om):


	xtt,xts=m[72:][:72]
	
	ytt,yts=om[72:],om[:72]
	
	ok.fit(xtt,ytt)
	print ok.score(xts,yts)


elfg=[]
elf=[]	
for i in range(len(m)/2):
	elfg.append(ok.predict(m[2*i]))
	elfg.append(ok.predict(m[2*i+1]))
for i in range(9):
	elf.append(0)
	elf.append(1)
ok.fit(acne[:17],elf[:17])
print ok.score(acne[15:],elf[15:])
for i in range(9):
	
	print ((elfg[36+i]-elfg[i])**2+(elfg[36+i+9]-elfg[i+9])**2+(elfg[36+i+18]-elfg[i+18])**2+(elfg[36+i+27]-elfg[i+27])**2)**0.5	
	