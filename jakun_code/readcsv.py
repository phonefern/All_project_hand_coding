import pandas as pd
import os
from glob import glob

path = 'D:\Code\HeatMapSerialRead\data\\16-01-2023*\\raw.csv'
files = glob(path)

print(files)
fixed=False

for file in files:
    df = pd.read_csv(file, encoding='utf-8', on_bad_lines='warn')
    print(df.shape)
    while(df.shape[0]%60!=0):
        # print(df.iloc[[-1]])
        # df = pd.concat(df.iloc[[-1]], ignore_index=True)
        df = df.append(df.iloc[[-1]], ignore_index=True)
        fixed = True
    if(fixed):
        df.to_csv(file, index=False, encoding='utf8')
    