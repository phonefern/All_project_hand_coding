import pandas as pd
import serial
import csv
import time
from datetime import datetime
import seaborn as sns
import os
from rich.console import Console
from rich.prompt import Prompt
import matplotlib.pyplot as plt
from pygame import mixer

console = Console()


def play_sound(path):
    # เรียกใช้งาน mixer และเล่นไฟล์เสียง
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    # หากไฟล์เสียงเป็น 'D:\Code\HeatMapSerialRead\heart-stop.mp3' หยุดการเล่นเสียงหลัง 0.3 วินาที
    if path == 'D:\Code\HeatMapSerialRead\heart-stop.mp3':
        time.sleep(0.3)
        mixer.music.stop()


# รายชื่อไฟล์เสียง
say = 'C:\\read_thermal\song\song_female_startread.mp3'
read_start = 'C:\\read_thermal\song\\5lastcall.mp3'
finish = 'C:\\read_thermal\song\\finish.mp3'
detail = 'C:\\read_thermal\song\\data.mp3'
heatmap = 'C:\\read_thermal\song\\heatmap.mp3'


class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r


class CsvWriter:
    def __init__(self, data, address, label):
        self.data = data
        self.address = address
        self.label = label

    def writeHead(self):
        head = ['TimeStamp']
        for index in range(768):
            head.append(index)
        head.append('Label')

        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(head)

    def process_raw(self):
        now = datetime.now()
        timenow = now.strftime("%d/%m/%Y-%H:%M:%S.%f")
        ls = [timenow]
        for x in self.data.split(','):
            ls.append(float(x.strip()))
        ls.append(self.label)
        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(ls)


# Function to collect data
def collect_data():
    play_sound(read_start)
    time.sleep(5)
    play_sound(say)
    
    readline = ReadLine(ser)
    count = 0
    combined_data = {}  # เพิ่มตัวแปรเพื่อเก็บข้อมูลรวม
    
    while count < 60:
        data_bytes = readline.readline()
        if not data_bytes:
            continue
        try:
            data_str = readline.readline().decode("UTF-8").strip()
            data_str = data_str.replace('[', '').replace(']', '')  # ลบ [ และ ] ออกจากข้อมูล
            now = datetime.now()
            time_now = now.strftime("%d/%m/%Y-%H:%M:%S.%f")
            CsvWriter(
                data=data_str, address=f"./data/{name}/raw.csv", label=label).process_raw()

            print("Data saved to CSV: {0}".format(count), end="\r")
            count += 1
            # ตรวจสอบว่า count ถึง 100 หรือไม่
            if count == 60:
                play_sound(finish)
                console.print("Data collection finished.", style="bold green")
                break
        except UnicodeDecodeError:
            console.print("Error decoding data. Skipping this line.")


# โปรแกรมหลัก
if __name__ == "__main__":
    # รับข้อมูลจากผู้ใช้
    Blackket = Prompt.ask("[bold red]Enter Your Blackket OR No Blackket[/bold red]")
    Subject = Prompt.ask("[bold red]Enter Your Name[/bold red]")
    gender = Prompt.ask("[bold red]Enter Your Gender[/bold red]")
    shape = Prompt.ask("[bold red]Enter Your Shape[/bold red]")
    label = Prompt.ask("[bold red]Enter Type Hand (Normal [0], Abnormal [1])[/bold red]")

    today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    name = f'{Blackket}--Gender-{gender}-Shape-{shape}-Type-{label}--{Subject}'
    name = today + '-' + name

    if name in os.listdir("./data"):
        console.print("File already exists.", style="bold red")
    else:
        os.makedirs(f"./data/{name}")
        CsvWriter(
            data=None, address=f"./data/{name}/raw.csv", label=label).writeHead()
        console.print("File created.", style="bold green")

    ser = serial.Serial('COM8', 115200, timeout=1)
    collect_data()

    # ประมวลผลและพล็อตข้อมูล
    df_path = f"./data/{name}"
    df = pd.read_csv(f'{df_path}/raw.csv')
    pixel_data = df.iloc[:, 1:769].values

    # เตรียมตัวแปรสำหรับการเก็บ Heatmap
    count = 0
    time_data = df['TimeStamp'].values
    heatmap_folder = f"./data/{name}/heatmaps"
    os.makedirs(heatmap_folder, exist_ok=True)

    # ทำ Heatmap จากข้อมูลพิกเซล
    for i, (array, time) in enumerate(zip(pixel_data, time_data)):
        frame2D = []
        for h in range(24):
            frame2D.append([])
            for w in range(32):
                t = array[h * 32 + w]
                frame2D[h].append(t)
        
        ax = sns.heatmap(frame2D, annot=True, fmt=".1f", cmap="coolwarm", linewidths=.1, yticklabels=False,
                         xticklabels=False, vmin=26, vmax=33, annot_kws={'size': 3.5})
        ax.set_title(f"Heatmap of MLX90640 at Time: {time}")
        
        heatmap_filename = f"{heatmap_folder}/Heatmap-{i + 1}.png"
        plt.savefig(heatmap_filename, dpi=300)
        plt.clf()

    print("Plot heatmap is successful.")
