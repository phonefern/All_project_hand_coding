import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('AGG')

while True:
    print('File :')
    for file in os.listdir("data"):
        print(f'- {file}')
    name = input("Enter name of file: ")
    if name not in os.listdir("data"):
        print("No File")
        continue
    else:
        break
data = pd.read_csv("./data/" + name + "/raw.csv", encoding="utf-8")
data.pop('Lable')
data.pop('TimeStamp')
count = 0
try:
    os.mkdir("./data/" + name + "/imagesNew/")
except:
    pass
for index, row in data.iterrows():
    print(f"Processing {index + 1}/{len(data)}")
    frame = []
    for valRow in row:
        frame.append(float(valRow))
    frame2D = []
    for h in range(24):
        frame2D.append([])
        for w in range(32):
            t = frame[h * 32 + w]
            frame2D[h].append(t)
    sns.heatmap(frame2D, annot=True, cmap="coolwarm", linewidths=.1, annot_kws={
                "size": 6}, yticklabels=False, xticklabels=False, vmin=25, vmax=29)
    plt.title("Heatmap of MLX90640 data: " + str(index) )
    plt.savefig("./data/" + name + "/imagesNew/" + str(count+1) + ".png")
    plt.close()
    count+=1
