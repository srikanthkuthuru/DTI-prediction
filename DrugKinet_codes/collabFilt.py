# Data
# Calculate Similarity matrices using compounds features and kinase features
import numpy as np
import pandas as pd
# Create the 3 matrices and write cell to load them easily
# Be in data/DrugKinet folder
file = pd.read_csv('ecfp4.csv')
compMat = file['ecfp4']
smilesMat = file['Smiles']
compMat = compMat.as_matrix()
T = [[int(i) for i in list(x)] for x in compMat]
compMat = np.asarray(T)
compBool = compMat
compMat = 2*compMat - 1

kinaseMat = pd.read_csv('../bowVecs5.csv')
kinases = pd.read_csv('Kinases.csv')
#Just changed
kinaseMat2 = pd.read_csv('../readinPy.csv')
kinaseFeat = np.zeros([246,5])
for i in range(len(kinaseMat)):
    a = kinaseMat.siRNA[i]
    ind = np.where(kinases == a)
    if(len(ind[0]) > 0):
        #kinaseFeat[ind[0],:] = np.array(kinaseMat.loc[i][1:]) #Change to 2: for readinPy.csv
        kinaseFeat[ind[0],:] = np.array(kinaseMat2.loc[i][2:]) #Change to 2: for readinPy.csv

dti = pd.read_csv('dti.csv')
dti = dti.as_matrix()    
dti = dti*2-1

from sklearn.metrics.pairwise import cosine_similarity as cosim
compSim = np.zeros([np.size(compMat,0), np.size(compMat,0)])
for i in range(np.size(compMat,0)):
    print(i)
    for j in range(np.size(compMat,0)):
        a1 = np.asarray(compBool[i,:]).astype(np.bool)
        a2 = np.asarray(compBool[j,:]).astype(np.bool)
        if(a1.sum() + a2.sum() == 0):
            compSim[i,j] = 1
        else:
            intersection = np.logical_and(a1, a2)
            compSim[i,j] = 2. * intersection.sum() / (a1.sum() + a2.sum())
    print(compSim[1,i])   


case = 'BoW';
if(case == 'BoW'):
    temp = np.sqrt(kinaseFeat);
    temp = np.matmul(temp,np.transpose(temp));
    kinaseSim = np.arccos(temp - 0.00001);   #Subtraction required for numerical consistency 
else:
    kinaseSim = cosim(kinaseFeat)
nt = len(kinaseSim)



#%% Collaborative Filtering - Matrix Completion
from copy import deepcopy
from sklearn import metrics

import matplotlib.pyplot as plt
from graphcodes import *
dsim = compSim;
tsim = kinaseSim
#tsim = (kinaseSim+1)/2
dknn = np.zeros(np.shape(dsim));
tknn = np.zeros(np.shape(tsim));
n,m = np.shape(dti)

#%%    
aucList = []; apcList = [];
nRow = np.size(dti,0)
nCol = np.size(dti,1)
testSize = 0.3
case = 'full';
for trial in range(1):
    if(case == 'fullRow'):
        mask = np.zeros([nRow, nCol])
        temp2 = np.random.rand(nRow)
        mask[temp2>testSize,:] = 1
    elif(case == 'fullCol'):
        mask = np.zeros([nRow, nCol])
        temp2 = np.random.rand(nCol)
        mask[:,temp2>testSize] = 1
    else:
        mask = np.random.rand(nRow, nCol)
        mask[mask<testSize] = 0
        mask[mask>=testSize] = 1

    #Find k-nearest neighbors in the training set    
    for i in range(n):
        temp=np.argsort(-dsim[i,:])
        dknn[i,temp[0:6]] = dsim[i,temp[0:6]]
    for j in range(m):
        temp=np.argsort(-tsim[j,:])
        #temp2 = temp2[temp];
        #temp = temp[temp2>testSize]; #Pick only the training set kinases
        tknn[j,temp[0:6]] = tsim[j,temp[0:6]]    
    
    dknn = (dknn + np.transpose(dknn))/2;
    tknn = (tknn + np.transpose(tknn))/2;





    ''' 
    # Completely new drugs
    mask = np.zeros([nRow, nCol])
    temp = np.random.rand(nRow)
    mask[temp>testSize,:] = 1
    '''
    
    dtiTemp = deepcopy(dti)
    dtiTemp[mask==0] = 0
    dtiTemp = dtiTemp.astype(float)
    print("dti-pos", sum(sum((dtiTemp==1))))
    
    
    #dti_hat = nuclear_norm_solve(dtiTemp, mask, mu=1.0)
    #dti_hat = svt_solve(dtiTemp, mask)
    deye = np.eye(nRow)
    teye = np.eye(nCol)
    dti_hat,U,V = graph_reg(dtiTemp, mask, deye, teye) 
    
    #dti_hat[dti_hat>0] = 1
    #dti_hat[dti_hat<=0] = -1
    dti_hat = dti_hat/np.max(abs(dti_hat))
    dti_hat = (dti_hat + 1)/2
    pred = dti_hat[mask == 0]
    true = dti[mask == 0]
    pred = np.ndarray.flatten(np.asarray(pred))
    #print("Accuracy =", sum(pred == true)/len(pred))
    #print("True Positive Rate", sum(pred[pred==true]==1)/sum(true==1))
    
    '''
    fpr, tpr, thresholds = metrics.roc_curve(true, pred, pos_label=1)
    plt.plot(fpr, tpr, color='darkorange',lw=2); 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()
    '''
    
    auc = metrics.roc_auc_score(true, pred)
    aupc = metrics.average_precision_score(true, pred)
    print("AUC score:",auc)
    
    apc = metrics.average_precision_score(true, pred)    
    print("APC score:",apc) 

    aucList.append(auc);
    apcList.append(apc);
#%%
print("Mean AUC score:",np.mean(aucList))
print("Mean APC score:",np.mean(apcList))

a = aucList
alpha = 95
ordered = np.sort(a)
lower = np.percentile(ordered, (100-alpha)/2)
upper = np.percentile(ordered, alpha+((100-alpha)/2))
print("lower:",lower)
print("upper:",upper)

a = apcList
alpha = 95
ordered = np.sort(a)
lower = np.percentile(ordered, (100-alpha)/2)
upper = np.percentile(ordered, alpha+((100-alpha)/2))
print("lower:",lower)
print("upper:",upper)


