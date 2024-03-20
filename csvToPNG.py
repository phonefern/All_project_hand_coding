import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# อ่านข้อมูลจากไฟล์ CSV
# df_path = r"C:\read_thermal\SumAllCase\Train_all.csv"
# df = pd.read_csv(f'{df_path}')
df_path = input("Enter the path to the CSV file: ")


df = pd.read_csv(f'{df_path}\\Data_simulate_fuck_off_0.4043_0.6759.csv')

# เลือกเฉพาะคอลัมน์ที่เกี่ยวข้องกับ pixel (65 pixel)
pixel_data = df.iloc[:, 1:65].values

# เลือกคอลัมน์ที่เกี่ยวข้องกับเวลา
time_data = df['TimeStamp'].values

count = 0

img_folder = f"{df_path}\\img_0.4_0.6"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)




for array, time in zip(pixel_data, time_data):
    count += 1
    frame2D = []
    for h in range(8):
        frame2D.append([])
        for w in range(8):
            t = array[h * 8 + w]
            frame2D[h].append(t)
    print("Plotting heatmap...")

    frame2D = list(map(list, zip(*frame2D)))

    # ax = sns.heatmap(frame2D, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.1, yticklabels=False, xticklabels=False, vmin=24,
    #                  vmax=29, annot_kws={'size': 6})
    ax = sns.heatmap(frame2D, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.1, yticklabels=False, xticklabels=False, vmin=21,
                        vmax=26, annot_kws={'size': 6})
    ax.set_title(f"Heatmap, at Time: {time}")
    plt.savefig(f"{img_folder}\\HMLabel-{count}.png", dpi=300)
    plt.clf()

print("Plot heatmap is success.")

