import serial
import time
import pandas as pd  # Import pandas library
import os
import csv
from datetime import date, datetime 
import seaborn as sns
import matplotlib.pyplot as plt
ser = serial.Serial('COM8', 115200, timeout=1)
time.sleep(2)

class ReadLine: # สร้าง class ชื่อ ReadLine
    def __init__(self, s): # สร้าง constructor โดยมี paremeter ชื่อ s
        self.buf = bytearray() # สร้างตัวเเปรชื่อ buf เป็น bytearray
        self.s = s  # กำหนดค่าให้กับตัวเเปร s

    def readline(self): # สร้าง method ชื่อ readline โดยมี paremeter self 
        i = self.buf.find(b"\n") # กำหนดค่าให้กับตัวเเปร i โดยให้ค้นหาตัวขึ้นบรรทัดใหม่
        if i >= 0: # ถ้า i มีค่ามากกว่าหรือเท่ากับ 0
            r = self.buf[:i+1] # กำหนดค่าให้กับตัวเเปร r โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
            self.buf = self.buf[i+1:] # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
            return r # ส่งค่า r กลับไป
        while True: # วนลูป
            i = max(1, min(2048, self.s.in_waiting)) # กำหนดค่าให้กับตัวเเปร i โดยให้เป็นค่าที่มากที่สุดระหว่าง 1 กับ 2048 
            data = self.s.read(i) # กำหนดค่าให้กับตัวเเปร data โดยให้เป็นค่าที่อ่านมาจาก serial port
            i = data.find(b"\n") # กำหนดค่าให้กับตัวเเปร i โดยให้ค้นหาตัวขึ้นบรรทัดใหม่
            if i >= 0: # ถ้า i มีค่ามากกว่าหรือเท่ากับ 0
                r = self.buf + data[:i+1] # กำหนดค่าให้กับตัวเเปร r โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
                self.buf[0:] = data[i+1:] # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นตัวอักษรตั้งเเต่ตัวเเรกจนถึงตัวขึ้นบรรทัดใหม่
                return r # ส่งค่า r กลับไป
            else:
                self.buf.extend(data) # กำหนดค่าให้กับตัวเเปร buff โดยให้เป็นค่าที่เก็บไว้ในตัวเเปร data

class Plot:
    def __init__(self, data, timeNow, name, roundLoop, save_directory="./plot_pic"):
        self.data = data
        self.time = int(datetime.timestamp(timeNow))
        self.roundLoop = roundLoop
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

    def processPlot(self):
        frame = [float(x.strip()) for x in self.data.split(',') if x]
        frame2D = [frame[h * 8:(h + 1) * 8] for h in range(8)]
        frame2D = list(map(list, zip(*frame2D)))
        sns.heatmap(frame2D, annot=True,fmt=".1f", cmap="coolwarm", linewidths=.1, annot_kws={"size": 6}, yticklabels=False, xticklabels=False, vmin=22, vmax=29)
        time_now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"Heatmap of AMG8833 data at Time: {time_now_str}"
        plt.title(title)
        filename = f"{self.time}_{self.roundLoop}.png"
        filepath = os.path.join(self.save_directory, filename)
        plt.savefig(filepath)
        plt.show()
        plt.close()

try:
    readline = ReadLine(ser)
    
    while True:
        line = readline.readline().decode().strip()
        plot_instance = Plot(line, datetime.now(), "firstset", 1, save_directory="./temp_01")
        
        plot_instance.processPlot()
        print('plot')

except KeyboardInterrupt:
    pass

