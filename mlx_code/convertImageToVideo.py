import cv2
import os
import re

image_folder = "C:\\HAND_GOD_GIVE\\PK_Readse\\data\\2-No-Blacket-Scenario\\1-Helthy-Weight\\07-10-2023\\07-10-2023 18.12_1-Helthy-Weight-Mork-0-Brachial-90\\img"
output_video_path = "C:\\HAND_GOD_GIVE\\PK_Readse\\data\\2-No-Blacket-Scenario\\1-Helthy-Weight\\07-10-2023\\07-10-2023 18.12_1-Helthy-Weight-Mork-0-Brachial-90\\img\\Mork_Br_90.mp4"
frame_rate = 1 # Adjust the frame rate as desired

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
