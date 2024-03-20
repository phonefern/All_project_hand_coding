import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import re

# อ่านข้อมูลจากไฟล์ CSV
df_path = r"C:\HAND_GOD_GIVE\PK_Readse\data\4-No-Blacket-With-Fan-Scenario\2-Overweight\16-10-2023\16-10-2023 16.58_2-Overweight-Formost-Shoulder-45-Label-1-With-Noise"
df = pd.read_csv(f'{df_path}\\rawData.csv')

# เลือกเฉพาะคอลัมน์ที่เกี่ยวข้องกับ pixel (768 pixel)
pixel_data = df.iloc[:, 1:769].values

# เลือกคอลัมน์ที่เกี่ยวข้องกับเวลา
# time_data = df['timestamp'].values
time_data = pd.to_datetime(df['timestamp'])

count = 0

for array, time in zip(pixel_data, time_data):
    count += 1
    frame2D = []
    for h in range(24):
        frame2D.append([])
        for w in range(32):
            t = array[h * 32 + w]
            frame2D[h].append(t)

    dt = time.strftime("%d-%m-%Y_%H:%M:%S")

    ax = sns.heatmap(frame2D, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.1, yticklabels=False, xticklabels=False, vmin=25,
                     vmax=32, annot_kws={'size': 3})
    ax.set_title(f"Heatmap, at Time: {dt}")
    plt.savefig(f"{df_path}\\img\\HMLabel-{count}.png", dpi=300)
    plt.clf()

print("Plot heatmap is success.")

image_folder = f"{df_path}\\img"
output_video_path = f"{df_path}\\0_1.mp4"
frame_rate = 1  # Adjust the frame rate as desired

# Get the list of image files in the folder and sort them numerically
image_files = sorted([file for file in os.listdir(image_folder) if file.endswith(".png")], key=lambda x: int(re.findall(r'\d+', x)[0]))

# Read the dimensions of the first image
first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
height, width, _ = first_image.shape

# Create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 video
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Loop through the image files and write each frame to the video
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)

# Release the VideoWriter and close the video file
video_writer.release()

print("Video conversion completed.")
