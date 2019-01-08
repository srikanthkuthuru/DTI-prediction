'''
Each cell below contains comments explaing the code in it.
1. Beginning of the file reads the compound feature matrix, kinase feature matrix and the dti matrix
2. Builds a simple collaborative filtering model for the dti matrix - perform training and testing
3. Builds Single task models using various ML classifiers like Logistic Regression, SVM, Neural network and KNN

'''
#%% Data Visualization
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
#%% Read matrices
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

kinaseMat = pd.read_csv('../bowVecs5.csv')
kinases = pd.read_csv('Kinases.csv')


#Just changed
kinaseMat2 = pd.read_csv('../readinPy.csv')
kinaseFeat = np.zeros([246,5])

for i in range(len(kinaseMat)):
    a = kinaseMat.siRNA[i]
    ind = np.where(kinases == a)
    if(len(ind[0]) > 0):
        kinaseFeat[ind[0],:] = np.array(kinaseMat.loc[i][1:]) #Change to 2: for readinPy.csv
        #kinaseFeat[ind[0],:] = np.array(kinaseMat2.loc[i][2:]) #Change to 2: for readinPy.csv
dti = pd.read_csv('dti.csv')
dti = dti.as_matrix()
dti = dti*2-1    

#%% Test for compound prediction
temp= compMat
compMat = kinaseFeat
dti = np.transpose(dti)

#%% Logistic Regression, SVM, Neural network and KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt

aucList = []; apcList = []; templist = [];
for trial in range(10):
    testSize = 0.3
    nRow = np.size(dti,0)
    nCol = np.size(dti,1)
    mask = np.zeros([nRow, nCol])
    temp = np.random.rand(nRow)
    mask[temp>testSize,:] = 1
    
    
    
    acc = []
    true = []; pred = []
    case = 'LR';
    for task in range(nCol):
        #print('Task:',task)
        if(case == 'LR'):
            model = LogisticRegression()
        elif(case == 'SVM'):
            model = SVC(kernel = 'linear', probability=True)
        elif(case == 'NN'):
            model = MLPClassifier(hidden_layer_sizes = (50), \
                                  activation = 'logistic', verbose = False, \
                                  learning_rate = 'adaptive')
        elif(case == 'KNN'):
            model = KNeighborsClassifier(n_neighbors=3, weights = 'distance')
        elif(case == 'RF'):
            model = RandomForestClassifier(n_estimators = 100, max_depth=5)
            
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
                templist = np.append(templist, metrics.roc_auc_score(ytest, 1-ypred[:,0]))
            except:
                #print('Error')
                continue
            
        
    
    
    true = np.ndarray.flatten(np.asarray(true))
    pred = np.ndarray.flatten(np.asarray(pred))
    
    #true = true[pred=='nan']
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
    print("AUC score:",auc)
    apc = metrics.average_precision_score(true, pred)    
    print("APC score:",apc)
    
    aucList.append(auc);
    apcList.append(apc);
    
    
    
    '''
    from sklearn.metrics import precision_recall_curve
    from sklearn.utils.fixes import signature
    precision, recall, _ = precision_recall_curve(true, pred)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(apc))
    '''
    
    
temp = np.random.choice(10,100)
aucList = np.array(aucList)
aucList = aucList[temp]
print("Mean AUC score:",np.mean(aucList))
print("CF:",np.percentile(aucList,[2.5,97.5]))

apcList = np.array(apcList)
apcList = apcList[temp]
print("Mean APC score:",np.mean(apcList))
print("CF:",np.percentile(apcList,[2.5,97.5]))

#%% MISC
temp= np.zeros([len(true),2]);
temp[:,0] = true;
temp[:,1] = pred;

