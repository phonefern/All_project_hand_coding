import serial
import time
import pandas as pd  # Import pandas library
import os
import csv
from datetime import date, datetime 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

ser = serial.Serial('COM8', 115200, timeout=1)
time.sleep(2)

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
            else:
                self.buf.extend(data)

class Plot:
    def __init__(self, data, timeNow, name, roundLoop, save_directory="./palm_out_nowave"):
        self.data = data
        self.time = int(datetime.timestamp(timeNow))
        self.roundLoop = roundLoop
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

    def processPlot(self):
        frame = []
        for x in self.data.split(','):
            if x == '':
                return
            frame.append(float(x.strip()))
        frame2D = []
        for h in range(8):
            frame2D.append([])
            for w in range(8):
                t = frame[h * 8 + w]
                frame2D[h].append(t)
        sns.heatmap(frame2D, annot=True, cmap="coolwarm", linewidths=.1, annot_kws={
                    "size": 6}, yticklabels=False, xticklabels=False, vmin=25, vmax=29)
        time_now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"Heatmap of MLX data at Time: {time_now_str}"
        plt.title(title)
        filename = f"{self.time}_{self.roundLoop}.png"
        filepath = os.path.join(self.save_directory, filename)
        plt.savefig(filepath)
        plt.show()
        plt.close()

try:
    readline = ReadLine(ser)
    
    while True:
        data_str = readline.readline().decode('utf-8').strip()
        data_str = data_str.replace('[', '').replace(']', '')
        plot_instance = Plot(data_str, datetime.now(), "firstset", 1, save_directory="./palm_out_nowave")
        
        plot_instance.processPlot()

except KeyboardInterrupt:
    pass
