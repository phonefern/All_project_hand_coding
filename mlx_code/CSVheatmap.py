import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
# อ่านข้อมูลจากไฟล์ CSV
df_path = "C:\HAND_GOD_GIVE\PK_Readse\data\\24-09-2023_17-20-13-Noblack--Gender-Male-Shape-Normal-Type-0--Team"
df = pd.read_csv(f'{df_path}\\raw.csv')

# เลือกเฉพาะคอลัมน์ที่เกี่ยวข้องกับ pixel (768 pixel)
pixel_data = df.iloc[:, 1:769].values

# เลือกคอลัมน์ที่เกี่ยวข้องกับเวลา
time_data = df['TimeStamp'].values

count = 0

for array, time in zip(pixel_data, time_data):
    count += 1
    frame2D = []
    for h in range(24):
        frame2D.append([])
        for w in range(32):
            t = array[h * 32 + w]
            frame2D[h].append(t)
    ax = sns.heatmap(frame2D, annot=False, cmap="coolwarm", linewidths=.1, yticklabels=False, xticklabels=False, vmin=24, vmax=29)
    ax.set_title(f"Heatmap of MLX8833 at Time: {time}")
    plt.savefig(f"C:\HAND_GOD_GIVE\PK_Readse\plot_pic\\PicTest-{count}.png", dpi=300)
    plt.show()
    plt.clf()

print("Plot heatmap is success.")
