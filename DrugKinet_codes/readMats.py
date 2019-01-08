#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:36:54 2018

@author: srikanthkuthuru
"""
#%% Remove this
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(compMat)
ydown = pca.transform(compMat)
import matplotlib.pyplot as plt
plt.plot(ydown[:,0], ydown[:,1], '.')
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title('PCA plot of drug features')


from sklearn.manifold import MDS
mds = MDS(n_components = 2, dissimilarity = 'precomputed')
ymds = mds.fit_transform(1-compSim)
plt.plot(ymds[:,0], ymds[:,1], '.')
plt.xlabel("Embedded dimension 1")
plt.ylabel("Embedded dimension 2")
plt.title('2-dimensional embedding of drug features with Metric MDS')


z = ymds[:,0]
y = ymds[:,1]
n = np.linspace(0,len(y)-1, len(y))
fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(int(txt), (z[i],y[i]))
ax.xlabel('Embedded dimension 1')
ax.ylabel('Embedded dimension 2')
ax.title('2-dimensional embedding of drug features with Metric MDS')
#%%
import numpy as np
import pandas as pd
#from matrix_completion import *
# Create the 3 matrices and write cell to load them easily
# Be in data/DrugKinet folder
compMat = pd.read_csv('ecfp4.csv')
compMat = compMat['ecfp4']
compMat = compMat.as_matrix()
T = [[int(i) for i in list(x)] for x in compMat]
compMat = np.asarray(T)
compMat = 2*compMat - 1

kinaseMat = pd.read_csv('../readinPy.csv')
kinases = pd.read_csv('Kinases.csv')
kinaseFeat = np.zeros([246,5])
for i in range(len(kinaseMat)):
    a = kinaseMat.siRNA[i]
    ind = np.where(kinases == a)
    if(len(ind[0]) > 0):
        kinaseFeat[ind[0],:] = np.array(kinaseMat.loc[i][2:])

dti = pd.read_csv('dti.csv')
dti = dti.as_matrix()
dti = dti*2-1    
#%% Collaborative Filtering - Matrix Completion
from copy import deepcopy
from sklearn import metrics
from graphcodes import *
import matplotlib.pyplot as plt
#Remove 20% elements from the matrix
nRow = np.size(dti,0)
nCol = np.size(dti,1)
testSize = 0.3
mask = np.random.rand(nRow, nCol)
mask[mask<testSize] = 0
mask[mask>=testSize] = 1

''' 
# Completely new drugs
mask = np.zeros([nRow, nCol])
temp = np.random.rand(nRow)
mask[temp>testSize,:] = 1
'''

dtiTemp = deepcopy(dti)
dtiTemp[mask==0] = 0
dtiTemp = dtiTemp.astype(float)



#dti_hat = nuclear_norm_solve(dtiTemp, mask, mu=1.0)
#dti_hat = svt_solve(dtiTemp, mask)
dti_hat = graph_reg(dtiTemp, mask, 1,1)

#dti_hat[dti_hat>0] = 1
#dti_hat[dti_hat<=0] = -1
dti_hat = dti_hat/np.max(abs(dti_hat))
dti_hat = (dti_hat + 1)/2
pred = dti_hat[mask == 0]
true = dti[mask == 0]
pred = np.ndarray.flatten(np.asarray(pred))
#print("Accuracy =", sum(pred == true)/len(pred))
#print("True Positive Rate", sum(pred[pred==true]==1)/sum(true==1))

fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
plt.plot(fpr, tpr, color='darkorange',lw=2); 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

auc = metrics.roc_auc_score(true, pred)
print("AUC score:",auc)
#%% Test for compuond prediction
temp= compMat
compMat = kinaseFeat
dti = np.transpose(dti)

testSize = 0.3
nRow = np.size(dti,0)
nCol = np.size(dti,1)
mask = np.zeros([nRow, nCol])
temp = np.random.rand(nRow)
mask[temp>testSize,:] = 1
#%% Logistic Regression, SVM, Neural network and KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt

acc = []
true = []; pred = []
case = 'LR';
for task in range(nCol):
    print('Task:',task)
    if(case == 'LR'):
        model = LogisticRegression()
    elif(case == 'SVM'):
        model = SVC(kernel = 'rbf', probability=True)
    elif(case == 'NN'):
        model = MLPClassifier(hidden_layer_sizes = (50), \
                              activation = 'logistic', verbose = False, \
                              learning_rate = 'adaptive')
    elif(case == 'KNN'):
        model = KNeighborsClassifier(n_neighbors=1, weights = 'distance')
    elif(case == 'RF'):
        model = RandomForestClassifier(n_estimators = 500, max_depth=10)
        
    testInds = (mask[:,task]==0)
    trainInds = (mask[:,task]==1)            
    if(case == 'SSLP'):
        model = LabelPropagation(kernel = 'knn', n_neighbors = 5)
        X = compMat
        y = (dti[:,task]+1)/2
        y[testInds] = -1
        model.fit(X,y)
        ypred = model.predict_proba(X[testInds,:])
        ytest = dti[testInds,task]
        pred = np.append(pred,1-ypred[:,0])
        true = np.append(true,ytest)
    

    else:
        X = compMat[trainInds,:]
        y = dti[trainInds,task]
        try:
            model.fit(X,y)
            Xtest = compMat[testInds,:]
            ytest = dti[testInds,task]
            ypred = model.predict_proba(Xtest)
            #print("Accuracy =", sum(ypred == ytest)/len(ypred))
            #acc.append(sum(ypred == ytest)/len(ypred))
            pred = np.append(pred,1-ypred[:,0])
            true = np.append(true,ytest)
            
        except:
            print('Error')
            continue
        
    

        
true = np.ndarray.flatten(np.asarray(true))
pred = np.ndarray.flatten(np.asarray(pred))
#true = true[pred=='nan']

fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
plt.plot(fpr, tpr, color='darkorange',lw=2); 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

auc = metrics.roc_auc_score(true, pred)
print("AUC score:",auc)
    







