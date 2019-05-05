#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:14:22 2019

@author: rishav
"""
import pandas as pd
import numpy as np
import os
import warnings
#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassPredictionError
#%%
warnings.filterwarnings('ignore')
os.chdir('/home/rishav/Test/drugsCom/Vader-Sentiment-Analysis')
#%%
#Working on Train Dataset Preprocessing and Reading
train_dataset=pd.read_csv('train_dataset_all.csv')
train_dataset=train_dataset[['rating', 'usefulCount', 'negative', 
                             'positive','neutral', 'compound', 'label']]
cat=pd.Categorical(train_dataset['label'])
train_dataset['label']=cat.codes
#%%
#Working on Splitting Train Dataset into 80% in train and 20% in test for
#cross valiation
train,test=train_test_split(train_dataset,test_size=0.2)
train_x,train_y=train.iloc[:,:-1],train.iloc[:,-1]
test_x,test_y=test.iloc[:,:-1],test.iloc[:,-1]
#%%
#Setting up KNN Classifier with 3 n-neighbours
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x,train_y)
#%%
#cross validation for accuracy with 5 K-Folds on train
scores_train = cross_val_score(knn, train_x, train_y, cv=5)
print('Accuracy : ',scores_train.mean()*100,'\n','Deviation : ',scores_train.std())
#%%
#score for accuracy with 5 K-Folds on test
scores_test = knn.score(test_x, test_y)
print('Accuracy : ',scores_test)
#%%
#Train Classification Reports
visualizer = ClassificationReport(knn,classes=cat.categories,support=True)
visualizer.score(test_x, test_y)
visualizer.poof()    
#%%
#Train Confusion Matrix
visualizer = ConfusionMatrix(knn,classes=cat.categories,label_encoder={i:j for i,j in enumerate(list(cat.categories))})
visualizer.score(test_x, test_y)
visualizer.poof()    
#%%
#Train Class Pred Bar graph
visualizer = ClassPredictionError(knn,classes=cat.categories)
visualizer.score(test_x, test_y)
visualizer.poof()    
#%%
#Working on Test Dataset Preprocessing and Reading
test_dataset=pd.read_csv('test_dataset_all.csv')
test_dataset=test_dataset[['rating', 'usefulCount', 'negative', 
                             'positive','neutral', 'compound', 'label']]
cat=pd.Categorical(test_dataset['label'])
test_dataset['label']=cat.codes
test_data_x,test_data_y=test_dataset.iloc[:,:-1],test_dataset.iloc[:,-1]
#%%
#cross validation for accuracy with 5 K-Folds on test
scores = knn.score(test_data_x, test_data_y)
print('Accuracy : ',scores*100)
#%%
#Test Classification Reports
visualizer = ClassificationReport(knn,classes=cat.categories,support=True)
visualizer.score(test_data_x, test_data_y)
visualizer.poof()    
#%%
#Test Confusion Matrix
visualizer = ConfusionMatrix(knn,classes=cat.categories,label_encoder={i:j for i,j in enumerate(list(cat.categories))})
visualizer.score(test_data_x, test_data_y)
visualizer.poof()    
#%%
#Test Class Pred Bar graph
visualizer = ClassPredictionError(knn,classes=cat.categories)
visualizer.score(test_data_x, test_data_y)
visualizer.poof()    







