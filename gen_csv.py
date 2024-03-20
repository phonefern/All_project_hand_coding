import pandas as pd
import serial
import csv
import time
from datetime import date, datetime
import seaborn as sns
import os
from rich.console import Console
from rich.prompt import Prompt
from pygame import mixer
console = Console()


def play(path):  # สร้างฟังก์ชันชื่อ play โดยมี paremeter ชื่อ path
    mixer.init()  # เริ่มใช้งาน mixer
    mixer.music.load(path)  # โหลดไฟล์เสียง
    mixer.music.play()  # เล่นไฟล์เสียง
    if (path == 'D:\Code\HeatMapSerialRead\heart-stop.mp3'):
        time.sleep(0.3)
        mixer.music.stop()


# path sound
say = 'C:\\read_thermal\song\song_female_startread.mp3'
readstart = 'C:\\read_thermal\song\\5lastcall.mp3'
finish = 'C:\\read_thermal\song\\finish.mp3'
detail = 'C:\\read_thermal\song\\data.mp3'

# play(finish)
# play(say)
# play(readstart)
# play(detail)


class ReadLine:  # สร้าง class ชื่อ ReadLine
    def __init__(self, s):  # สร้าง constructor โดยมี paremeter ชื่อ s
        self.buf = bytearray()  # สร้างตัวเเปรชื่อ buf เป็น bytearray
        self.s = s  # กำหนดค่าให้กับตัวเเปร s

    def readline(self):  # สร้าง method ชื่อ readline โดยมี paremeter self
        # กำหนดค่าให้กับตัวเเปร i โดยให้ค้นหาตัวขึ้นบรรทัดใหม่
        i = self.buf.find(b"\n")
        if i >= 0:  # ถ้า i มีค่ามากกว่าหรือเท่ากับ 0
            # กำหนดค่าให้กับตัวเเปร r โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
            r = self.buf[:i+1]
            # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
            self.buf = self.buf[i+1:]
            return r  # ส่งค่า r กลับไป
        while True:  # วนลูป
            # กำหนดค่าให้กับตัวเเปร i โดยให้เป็นค่าที่มากที่สุดระหว่าง 1 กับ 2048
            i = max(1, min(2048, self.s.in_waiting))
            # กำหนดค่าให้กับตัวเเปร data โดยให้เป็นค่าที่อ่านมาจาก serial port
            data = self.s.read(i)
            # กำหนดค่าให้กับตัวเเปร i โดยให้ค้นหาตัวขึ้นบรรทัดใหม่
            i = data.find(b"\n")
            if i >= 0:  # ถ้า i มีค่ามากกว่าหรือเท่ากับ 0
                # กำหนดค่าให้กับตัวเเปร r โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
                r = self.buf + data[:i+1]
                # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
                self.buf[0:] = data[i+1:]
                return r  # ส่งค่า r กลับไป
            else:
                # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นค่าที่เก็บไว้ในตัวเเปร data
                self.buf.extend(data)


class csvWrite:  # This is a class to write data from sensor to csv file
    def __init__(self, data, address, label):
        self.data = data
        self.address = address
        self.label = label

    def writeHead(self):
        head = ['TimeStamp']
        for index in range(64):
            head.append(index)
        head.append('Label')

        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(head)
            write.close()

    def processRaw(self):
        now = datetime.now()
        timenow = now.strftime("%d/%m/%Y-%H:%M:%S.%f")
        # frame = [float(x.strip()) for x in self.data.split(',')]
        # dataToCsv = np.array([timenow,frame,self.label])
        # df = pd.DataFrame(dataToCsv)
        # df.T.to_csv(self.address, mode="a", header=False, index=False)

        ls = []
        ls = [timenow]
        # วนลูปเพื่อเก็บข้อมูลลงใน list โดยให้ตัด , ออกจากข้อมูล
        for x in self.data.split(','):
            # เก็บข้อมูลลงใน list โดยตัด \n ออกจากข้อมูล
            ls.append(float(x.strip()))
        ls.append(self.label)  # เพิ่ม label ลงใน list

        with open(self.address, 'a+', newline='') as write:  # เขียนข้อมูลลงในไฟล์ csv
            # สร้างตัวเเปร writer เพื่อเขียนข้อมูลลงในไฟล์ csv
            writer = csv.writer(write)
            # เขียนข้อมูลลงในไฟล์ csv โดยให้ข้อมูลเป็น list ที่เก็บไว้ในตัวเเปร ls
            writer.writerow(ls)
            write.close()  # ปิดการเขียนไฟล์ csv


play(detail)
# rounds = Prompt.ask("[bold cyan]Enter rounds[/bold cyan]")
Blackket = Prompt.ask(
    "[bold red] Enter Your Blackket OR No Blackket [/bold red]")
Subject = Prompt.ask("[bold red] Enter Your Name [/bold red]")
gender = Prompt.ask("[bold red] Enter Your Gender [/bold red]")
shape = Prompt.ask("[bold red] Enter Your Shape [/bold red]")
label = Prompt.ask(
    "[bold red] Enter Type Hand (Normal [0], Abnormal [1]) [/bold red]")

# วันที่ปัจจุบัน ตั้งชื่อโฟลเดอร์
today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# ตั้งชื่อโฟลเดอร์
name = f'{Blackket}--Gender-{gender}-Shape-{shape}-Type-{label}--{Subject}'
name = today + '-' + name

# กำหนดค่าให้กับตัวเเปร ser โดยให้เป็น serial port COM8 ที่ความเร็ว 115200
ser = serial.Serial('COM11', 115200, timeout=1)

# ตรวจสอบว่าโฟลเดอร์ name ยังไม่มีอยู่ใน "data"
# ถ้ามีอยู่แล้ว ให้แสดงข้อความว่า "File already exists."
if name in os.listdir("./data"):
    console.print("File already exists.", style="bold red")
else:

    os.makedirs(f"./data/{name}")  # ถ้ายังไม่มี ให้สร้างโฟลเดอร์ใหม่
    # สร้าง CSV ไฟล์และเขียนหัวคอลัมน์
    # สร้างไฟล์ raw.csv ในโฟลเดอร์ name
    csvWrite(
        data=None, address=f"./data/{name}/raw.csv", label=label).writeHead()
    console.print("File created.", style="bold green")

play(readstart)
time.sleep(5)
play(say)

readline = ReadLine(ser)
count = 0

while count < 250:  # Continue collecting data until count reaches 900
    data = readline.readline().decode()  # Read data from the serial port
    if data:
        # Insert timestamp
        now = datetime.now()
        timenow = now.strftime("%d/%m/%Y-%H:%M:%S.%f")

        # Save data to CSV
        csvWrite(
            data=data, address=f"./data/{name}/raw.csv", label=label).processRaw()
        # Print the count on the same line
        print("Data saved to CSV: {0}".format(count), end="\r")
        count += 1  # Increment the count variable by 1 for each data entry


play(finish)
# Print the success message
console.print("Data collection finished.", style="bold green")

