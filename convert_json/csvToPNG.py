import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('DB_jsonToCSV_testData4.csv')

# เลือกเฉพาะคอลัมน์ที่เกี่ยวข้องกับ pixel 
pixel_data = df.iloc[:, 1:65].values

# เลือกคอลัมน์ที่เกี่ยวข้องกับเวลา
time_data = df['timestamp'].values

count = 0

for array, time in zip(pixel_data, time_data):
    count += 1
    frame2D = []
    for h in range(8):
        frame2D.append([])
        for w in range(8):
            t = array[h * 8 + w]
            frame2D[h].append(t)
    ax = sns.heatmap(frame2D, annot=True, cmap="coolwarm", linewidths=.1, annot_kws={
                "size": 6}, yticklabels=False, xticklabels=False, vmin=22, vmax=30)
    ax.set_title(f"Heatmap of : {count} , Time: {time}")

    # # เพิ่มเวลาในตำแหน่งมุมของ Heatmap
    # ax.text(0, -1, f'Time: {time}', ha='left', va='bottom', fontsize=8, color='black')

    plt.savefig(f"D:\picture_heatmap\label-{count}.png")
    plt.show()
    plt.clf()

print("Plot heatmap is success.")
