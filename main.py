#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:54:14 2019

@author: rishav
"""
import pandas as pd
import numpy as np
import os
import copy as cp
import vaderSentiment.vaderSentiment as vd
import warnings
import dask.dataframe as dd
#%%
warnings.filterwarnings('ignore')
os.chdir('/home/rishav/Test/drugsCom/Vader-Sentiment-Analysis')
train_dataset=pd.read_table('drugsComTrain_raw.tsv')
test_dataset=pd.read_table('drugsComTest_raw.tsv')
#%%
train_dataset.set_index('Unnamed: 0',inplace=True)
test_dataset.set_index('Unnamed: 0',inplace=True)
train_dataset.drop(columns=['drugName','date'],inplace=True)
test_dataset.drop(columns=['drugName','date'],inplace=True)
train_dataset['negative']=np.nan
train_dataset['positive']=np.nan
train_dataset['neutral']=np.nan
train_dataset['compound']=np.nan
train_dataset['label']=np.nan
test_dataset['negative']=np.nan
test_dataset['positive']=np.nan
test_dataset['neutral']=np.nan
test_dataset['compound']=np.nan
test_dataset['label']=np.nan
train_dataset.sort_index(inplace=True)
train_dataset.reset_index(inplace=True)
train_dataset.drop(columns=['Unnamed: 0'],inplace=True)
test_dataset.reset_index(inplace=True)
test_dataset.sort_index(inplace=True)
test_dataset.drop(columns=['Unnamed: 0'],inplace=True)
#%%
def function(temp):
    analyser=vd.SentimentIntensityAnalyzer()
    
    def filter_data_label(c):
        if c>=0.05:
            return('Positive')
        elif c<=-0.05:
            return('Negative')
        else:
            return('Neutral')

    for i in temp.iterrows():
        analysis=analyser.polarity_scores(i[1][1])
        
        temp['negative'].loc[i[0]]=analysis['neg']
        temp['positive'].loc[i[0]]=analysis['pos']
        temp['neutral'].loc[i[0]]=analysis['neu']
        temp['compound'].loc[i[0]]=analysis['compound']
        temp['label'].loc[i[0]]=filter_data_label(analysis['compound'])
        print(i[0])

    return(temp)

#%%

train_dataset_10000=function(train_dataset[:10001])
train_dataset_10000.to_csv('train_dataset_10000.csv',index=False)

#%%
from dask.distributed import Client
client = Client(processes=4)
client
temp_train=dd.read_csv('drugsComTrain_raw.tsv',blocksize=10000,sep='\t')
temp_train=temp_train.loc[:,['condition', 'review', 'rating', 'usefulCount']]
temp_train['negative']=np.nan
temp_train['positive']=np.nan
temp_train['neutral']=np.nan
temp_train['compound']=np.nan
temp_train['label']=np.nan
temp=temp_train.map_partitions(lambda df:function(df))
temp = client.persist(temp)
type(temp)
temp.head()

