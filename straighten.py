import numpy as np
import pandas as pd
from os import listdir

activities = ['a01','a02','a03','a04','a05','a06']
column_names = ['bax','bay','baz','bgx','bgy','bgz','gax','gay','gaz']

def load_csv(name):
    dataFrame = pd.read_csv(name,skiprows=1, names = ['index','bax','bay','baz','bgx','bgy','bgz','gax','gay','gaz'])
    dataFrame = dataFrame.drop(['index'],axis=1)
    return dataFrame

def straighten(df):
    sdf = pd.DataFrame()
    for index,row in df.iterrows():
        for key,value in row.iteritems():
            sdf[key+str(index)] = [value]
    return sdf

#Train
for a in activities:
    csvs = listdir('train/'+a+'/')
    for filename in csvs:
        name = 'train/'+a+'/'+filename
        print(name)
        df = load_csv(name)
        sdf = straighten(df)
        sdf.to_csv(name,index=False)

#Validation
for a in activities:
    csvs = listdir('validation/'+a+'/')
    for filename in csvs:
        name = 'validation/'+a+'/'+filename
        print(name)
        df = load_csv(name)
        sdf = straighten(df)
        sdf.to_csv(name,index=False)

#Test
for a in activities:
    csvs = listdir('test/')
    for filename in csvs:
        name = 'test/'+filename
        print(name)
        df = load_csv(name)
        sdf = straighten(df)
        sdf.to_csv(name,index=False)
