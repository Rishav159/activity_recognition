import numpy as np
import pandas as pd
from os import listdir

activities = ['a01','a02','a03','a04','a05','a06']
column_names = ['bax0','bay0','baz0','bgx0','bgy0','bgz0','gax0','gay0','gaz0','bax1','bay1','baz1','bgx1','bgy1','bgz1','gax1','gay1','gaz1','bax2','bay2','baz2','bgx2',
                'bgy2','bgz2','gax2','gay2','gaz2','bax3','bay3','baz3','bgx3','bgy3','bgz3','gax3','gay3','gaz3','bax4','bay4','baz4','bgx4','bgy4','bgz4','gax4','gay4','gaz4','activity']

def load_csv(name):
    dataFrame = pd.read_csv(name)
    return dataFrame

#Train
train_df = pd.DataFrame(columns = column_names)
for a in activities:
    csvs = listdir('train/'+a+'/')
    for filename in csvs:
        df = pd.read_csv('train/'+a+'/'+filename)
        train_df = pd.concat([train_df,df])
    train_df = train_df.fillna(int(a[2]))
train_df.to_csv('train.csv',index=False)
print(train_df)

#Validation
train_df = pd.DataFrame(columns = column_names)
for a in activities:
    csvs = listdir('validation/'+a+'/')
    for filename in csvs:
        df = pd.read_csv('validation/'+a+'/'+filename)
        train_df = pd.concat([train_df,df])
    train_df = train_df.fillna(int(a[2]))
train_df.to_csv('validation.csv',index=False)
print(train_df)
