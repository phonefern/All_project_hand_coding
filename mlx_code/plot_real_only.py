import serial
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import re

ser = serial.Serial('COM8', 115200, timeout=1)
time.sleep(2)

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

class Plot:
    def __init__(self):
        pass

    def processPlot(self, data):
        # ใช้ regular expression เพื่อหาข้อมูลที่อยู่ในวงเล็บสี่เหลี่ยม
        data_inside_brackets = re.search(r'\[([^\]]+)', data)
        if data_inside_brackets:
            data_inside_brackets = data_inside_brackets.group(1)
            frame = [float(x.strip()) for x in data_inside_brackets.split(',') if x]
            frame2D = [frame[h * 8:(h + 1) * 8] for h in range(8)]
            sns.heatmap(frame2D, annot=True, cmap="coolwarm", linewidths=0.1, annot_kws={
                        "size": 6}, yticklabels=False, xticklabels=False, vmin=22, vmax=29)
            plt.title("Real-time Heatmap of MLX90640 Data")
            plt.show()
            plt.close()

try:
    readline = ReadLine(ser)
    plotter = Plot()

    while True:
        line = readline.readline().decode().strip()
        plotter.processPlot(line)

except Exception as e:
    print(f"Error: {e}")

print("Get Data Success.")
