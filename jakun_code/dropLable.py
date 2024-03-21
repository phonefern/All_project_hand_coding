import pandas as pd
import os
from time import sleep
import glob

files = os.path.join(f"D:\Code\HeatMapSerialRead\data\*case*-lo*\*raw.csv")
files = glob.glob(files)

for i in files:
    data = pd.read_csv(i,encoding='utf8',on_bad_lines='skip')
    print(i)
    if 'Lable' in data:
        print('Drop Lable')
        data.drop('Lable',axis=1,inplace=True)
        data.to_csv(i, index=False)
        print('✔')
        # sleep(5)
        # print(data)