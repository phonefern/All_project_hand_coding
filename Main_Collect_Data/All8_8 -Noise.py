import pandas as pd
import serial
import csv
import time
from datetime import datetime
import seaborn as sns
import os
from rich.console import Console
from rich.prompt import Prompt
from pygame import mixer
import matplotlib.pyplot as plt

console = Console()
mixer.init()

# Function to play sound


def play_sound(path):
    mixer.music.load(path)
    mixer.music.play()
    if path == 'D:\Code\HeatMapSerialRead\heart-stop.mp3':
        time.sleep(0.3)
        mixer.music.stop()


# Define sound paths
say = 'C:\\read_thermal\song\song_female_startread.mp3'
read_start = 'C:\\read_thermal\song\\5lastcall.mp3'
finishtt = 'C:\\read_thermal\song\\Finish.mp3'
detail = 'C:\\read_thermal\song\\data.mp3'
heatmap = 'C:\\read_thermal\song\\heatmap.mp3'
frame = 'C:\\read_thermal\song\\10Flame.mp3'
frameleft = 'C:\\read_thermal\song\\FrameLeaft.mp3'
# Function to read lines from serial
class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i + 1]
            self.buf = self.buf[i + 1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i + 1]
                self.buf[0:] = data[i + 1:]
                return r
            else: 
                self.buf.extend(data)

# Class to write data to CSV


class CsvWriter:
    def __init__(self, data, address, label):
        self.data = data
        self.address = address
        self.label = label

    def write_header(self):
        head = ['TimeStamp']
        for index in range(64):
            head.append(index)
        head.append('Label')

        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(head)

    def process_raw(self):
        now = datetime.now()
        time_now = now.strftime("%d/%m/%Y-%H:%M:%S.%f")
        ls = [time_now]
        for x in self.data.split(','):
            ls.append(float(x.strip()))
        ls.append(self.label)

        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(ls)

# Function to perform data collection


def collect_data():
    play_sound(read_start)
    time.sleep(9)
    play_sound(say)

    readline = ReadLine(ser)
    count = 0
    try:
        while count < 360:
            data = readline.readline().decode()
            if data: # If data is not empty
                now = datetime.now()
                time_now = now.strftime("%d/%m/%Y-%H:%M:%S.%f")
                CsvWriter(
                    data=data, address=f"./data_v2/{name}/raw.csv", label=label).process_raw()

                print("Data saved to CSV: {0}".format(count), end="\r")
                count += 1
                if count == 330:
                    play_sound(frameleft)
                elif count == 359:
                    play_sound(finishtt)
                    time.sleep(3)

    except KeyboardInterrupt:
        # Handle Ctrl+C (KeyboardInterrupt)
        print("\nData collection interrupted.")
    except Exception as e:
        print(f"Error: {e}")

    
    console.print("Data collection finished.", style="bold green")



# Main program
if __name__ == "__main__":
    # Prompt for user input
    case = Prompt.ask("[bold red]Enter Your Case [Laydown] [Bk] [NoBk] [FNoBk] [FBk] [/bold red]")
    Subject = Prompt.ask("[bold red]Enter Your Name[/bold red]")
    Post = Prompt.ask("[bold red]Enter Your Post [Br90] [Br45] [Sh90] [Sh45] [/bold red]")
    shape = Prompt.ask("[bold red]Enter Your Shape [Thin] [Normal] [Over] [Fat] [/bold red]")
    label = Prompt.ask("[bold red]Enter Type Hand (Normal [0], Abnormal [1])[/bold red]")

    today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    name = f'Case--{case}--{Post}--{shape}-lable-{label}--{Subject}'
    name = today + '-' + name

    if name in os.listdir("./data_v2"):
        console.print("File already exists.", style="bold red")
    else:
        os.makedirs(f"./data_v2/{name}")
        CsvWriter(
            data=None, address=f"./data_v2/{name}/raw.csv", label=label).write_header()
        console.print("File created.", style="bold green")

    ser = serial.Serial('COM8', 115200, timeout=1)
    collect_data()
    # Process and plot data
    df_path = f"./data_v2/{name}"
    df = pd.read_csv(f'{df_path}/raw.csv')
    pixel_data = df.iloc[:, 1:65].values
    
# เลือกคอลัมน์ที่เกี่ยวข้องกับเวลา
time_data = df['TimeStamp'].values

count = 0
    # หลังจากสร้างโฟลเดอร์ "name" แล้ว
heatmap_folder = f"./data/{name}/heatmap"
os.makedirs(heatmap_folder)

for array, time in zip(pixel_data, time_data):
        count += 1
        frame2D = []
        for h in range(8):
            frame2D.append([])
            for w in range(8):
                t = array[h * 8 + w]
                frame2D[h].append(t)

        # Transpose the frame2D array to make it vertical (portrait mode)
        frame2D = list(map(list, zip(*frame2D)))

        ax = sns.heatmap(frame2D, annot=True, cmap="coolwarm", fmt=".1f",
                         linewidths=.1, yticklabels=False, xticklabels=False, vmin=24, vmax=29)
        ax.set_title(f"Heatmap of AMG8833 at Time: {time}")
        heatmap_file_path = f"{heatmap_folder}/PicTest-{count}.png"
        plt.savefig(heatmap_file_path, dpi=300)
        plt.clf()

print("Plot heatmap is success.")
play_sound(heatmap)