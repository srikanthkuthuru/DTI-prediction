#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:01:55 2018

@author: srikanthkuthuru
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
acc = []
true = []; pred = []

dti = (dti+1)/2

nRow = np.size(dti,0)
nCol = np.size(dti,1)
testSize = 0.3
aucList = []; apcList = [];
for i in range(1):
    a = np.arange(nRow)
    temp = np.random.rand(nRow)
    trainInds = a[temp>testSize]
    testInds = a[temp<testSize]
    
    
    Xtr = compMat[trainInds,:]
    ytr = dti[trainInds,:]
    
    Xts = compMat[testInds,:]
    yts = dti[testInds,:]
    
    import keras.backend as K
    
    def multitask_loss(y_true, y_pred):
        # Avoid divide by 0
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Multi-task loss
        return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))
    
    
        
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Activation
    from sklearn import metrics
    batch_size = 30
    epochs = 200
    
    
    
    model = Sequential()
    model.add(Dense(100,input_shape=(np.size(Xtr,1),)))
    model.add(Activation('sigmoid'))
    
    
    model.add(Dense(nCol))
    model.add(Activation('sigmoid'))
    
    
    model.compile(loss=multitask_loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    model.fit(Xtr, ytr,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(Xts, yts))
    
    ypred = model.predict(Xts)
    true = np.ndarray.flatten(np.asarray(yts))
    pred = np.ndarray.flatten(np.asarray(ypred))
    
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