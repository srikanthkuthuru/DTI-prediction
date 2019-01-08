'''
Author: Srikanth Kuthuru, Rice University
1.Calculate Similarity matrices using compounds features and kinase features
2. This create 3 matrices named dti,dsim,tsim
3. Train and test yamanishi's code based on bipartite networks 
- didn't get expected results - debug ? any wrong?

'''

#%% 
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
kinaseFeat = np.zeros([246,5])
for i in range(len(kinaseMat)):
    a = kinaseMat.siRNA[i]
    ind = np.where(kinases == a)
    if(len(ind[0]) > 0):
        kinaseFeat[ind[0],:] = np.array(kinaseMat.loc[i][1:]) #Change to 2: for readinPy.csv

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
kinaseSim = cosim(kinaseFeat)
nt = len(kinaseSim)
#%% Clustering
import matplotlib.pyplot as plt 
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
d = pdist(compMat, metric = 'hamming')
from scipy.cluster.hierarchy import linkage
z = linkage(d, method = 'complete')
from scipy.cluster.hierarchy import dendrogram
plt.figure()
dn = dendrogram(z)


#import scipy
#scipy.io.savemat('compMat.mat', mdict={'compMat': compMat})
#%%
nRow = np.size(dti,0)
nCol = np.size(dti,1)
testSize = 0.3
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import MDS
#Training
a = np.arange(nRow)
temp = np.random.rand(nRow)
trainInds = a[temp>testSize]
testInds = a[temp<testSize]
nd = len(trainInds)
dtiTrain = dti[trainInds,:]
d = np.zeros([nd+nt, nd+nt]) #Adjacency matrix for the bipartite graph
d[:nd,:nd] = np.eye(nd)
d[nd:,nd:] = np.eye(nt)
d[:nd,nd:] = dtiTrain
d[nd:,:nd] = np.transpose(dtiTrain)


G=nx.from_numpy_matrix(d) #Graph from adjacency matrix
h = 1
#Create matrix K
K = np.zeros([nd+nt,nd+nt])
lenDict=nx.all_pairs_shortest_path_length(G)
count=0
for temp in lenDict:
    print(count)
    for j in range(len(temp[1])):
        if j in temp[1]:
            K[count,j] = np.exp(-temp[1][j]**2/h**2)
    count = count + 1

# Mean Centering
n1 = np.ones([len(K), len(K)])*1.0/len(K)
K2 = K - np.dot(n1,K) - np.dot(K,n1) + np.linalg.multi_dot([n1,K,n1])
[t1,t2] = np.linalg.eig(K2)
print(np.min(np.real(t1)))
K3 = K2 - np.min(np.real(t1))*np.eye(len(K2))    
[t1,t2] = np.linalg.eig(K3)
print(np.min(np.real(t1)))
U = np.real(np.matmul(t2,np.diag(np.sqrt(t1))))

# Apply MDS
q = 50
Ksym = (K + np.transpose(K))/2
mds = MDS(n_components = q, metric = True, dissimilarity = 'precomputed')
U = mds.fit_transform(Ksym)





#pca = PCA(n_components=q)
#U = U[:,:q]

#For Drugs
temp = (compSim[trainInds,:])[:,trainInds] + 0.1*np.eye(nd)
A = np.linalg.multi_dot([np.linalg.inv(temp), U[:nd,:], np.transpose(U[:nd,:]), np.linalg.inv(temp)]) #A = UU^T
[t1,t2] = np.linalg.eig(A)
W = np.real(np.matmul(t2,np.diag(np.sqrt(t1))))[:,:q]


'''
#For targets
temp2 = KinaseSim
A2 = np.linalg.multi_dot([np.linalg.inv(temp2), U[nd:,:], np.transpose(U[nd:,:]), np.linalg.inv(temp2)]) #A = UU^T
[t1,t2] = np.linalg.eig(A2)
W2 = np.real(np.matmul(t2,np.diag(np.sqrt(t1))))[:,:q] 
'''

count = 0
pharmaPred = np.zeros([len(testInds),q])
for ind in testInds:
    l = compSim[ind,trainInds]
    pharmaPred[count,:] = np.matmul(l, W) 
    count+=1
finMat = np.matmul(pharmaPred,np.transpose(U[nd:,:]))

pred = np.ndarray.flatten(finMat)
true = np.ndarray.flatten(dti[testInds,:])
#Select top 1%
inds = np.where(pred > np.percentile(pred, 95))

fpr, tpr, thresholds = metrics.roc_curve(true[inds], pred[inds], pos_label=1)
plt.plot(fpr, tpr, color='darkorange',lw=2)



plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()
auc = metrics.roc_auc_score(true[inds], pred[inds])
print('AUC=',auc)
    
    
import scipy
scipy.io.savemat('FinalMats/dtimat.mat', mdict={'dti': dti})
scipy.io.savemat('FinalMats/dsim.mat', mdict={'dsim': compSim})
scipy.io.savemat('FinalMats/tsim.mat', mdict={'tsim': kinaseSim})

    