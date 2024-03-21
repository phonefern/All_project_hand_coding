import glob
import os
import pandas as pd
from time import sleep

files = os.path.join(f"D:\Code\HeatMapSerialRead\data\*case*-lo*\*raw.csv")
files = glob.glob(files)


# print(files)

# ignore = ['07-12-2022','08-12-2022','09-12-2022']
# target = files

for ig in ignore:
    for index,file in enumerate(target):
        if ig in file:
            target[index] = ''
# print(*target,sep='\n')

for i in target:
    if i:
        print(i)
        # data = pd.read_csv(i,encoding='utf8',on_bad_lines='skip')
        # label = data['Label'].unique()
        # print(label)
        # sleep(50)
        if(label==2):
            print('Changing label to 1')
            data['Label'] = 1
        elif(label==3):
            print('Changing label to 2')
            data['Label'] = 2
        data.to_csv(i, index=False)
        print('✔')
        sleep(3)
        # print(data)