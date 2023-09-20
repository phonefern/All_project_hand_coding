import csv
import os
import time
from datetime import date, datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import serial.tools.list_ports
from rich.console import Console
from rich.prompt import Prompt
from win10toast import ToastNotifier
from pygame import mixer

console = Console()
sns.set_style({'font.family': 'Times New Roman'})
toast = ToastNotifier()
# matplotlib.use('Agg')


def play(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    if (path == 'D:\Code\HeatMapSerialRead\heart-stop.mp3'):
        time.sleep(0.3)
        mixer.music.stop()


# path sound
clap = 'D:\Code\HeatMapSerialRead\clap.mp3'
signal = 'D:\Code\HeatMapSerialRead\heart-stop.mp3'
start = 'D:\Code\HeatMapSerialRead\start.mp3'
countdown = 'D:\Code\HeatMapSerialRead\countdown.mp3'
tenframeleft = 'D:\\Code\\HeatMapSerialRead\\10frameleft.mp3'
rest = 'D:\\Code\\HeatMapSerialRead\\rest.mp3'
willstart = 'D:\\Code\\HeatMapSerialRead\\willstart.wav'
finish = 'D:\\Code\\HeatMapSerialRead\\finish.mp3'

# play(finish)
# play(willstart)
# play(clap)
play(signal)


def showToast():
    toast.show_toast(
        "Susscess", "Scraping completed.", duration=1, threaded=True
    )


class detectSerial:
    def __init__(self):
        self.serialList = []

    def get_serial_list(self):
        for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
            self.serialList.append(port)
        if len(self.serialList) > 1:
            console.print("Serial ports found:")
            for i in range(len(self.serialList)):
                console.print(str(i + 1) + ": " + self.serialList[i])
            return Prompt.ask("[bold cyan]Select serial port[/bold cyan] ").lower()
        elif len(self.serialList) == 1:
            return self.serialList[0]
        else:
            console.print("No serial ports found.", style="bold red")
            exit()


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


class plot:
    def __init__(self, data, timeNow, name, roundLoop):
        self.data = data
        self.time = int(datetime.timestamp(timeNow))
        self.roundLoop = roundLoop
        self.address = "./data/" + name + "/images/" + \
            str(self.time) + str(self.roundLoop) + ".png"

    def processPlot(self):
        frame = []
        # print(self.data.split(','))
        for x in self.data.split(','):
            if x == '':
                return
            frame.append(float(x.strip()))
        # print(frame)
        frame2D = []
        for h in range(24):
            frame2D.append([])
            for w in range(32):
                t = frame[h * 32 + w]
                frame2D[h].append(t)
        sns.heatmap(frame2D, annot=True, cmap="coolwarm", linewidths=.1, annot_kws={
                    "size": 6}, yticklabels=False, xticklabels=False, vmin=25, vmax=29)
        plt.title("Heatmap of MLX90640 data: " +
                  str(self.time) + str(self.roundLoop))
        plt.savefig(self.address)
        plt.show()
        plt.close()


class csvWrite: # This is a class to write data from sensor to csv file
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
        for x in self.data.split(','):
            ls.append(float(x.strip()))
        ls.append(self.label)

        with open(self.address, 'a+', newline='') as write:
            writer = csv.writer(write)
            writer.writerow(ls)
            write.close()

    def processCount(self):
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0
        error = 0
        total = 0
        for i in self.data.split(','):
            count = float(i)
            if count >= 38 and count < 40:
                a += 1
            elif count >= 36 and count < 38:
                b += 1
            elif count >= 34 and count < 36:
                c += 1
            elif count >= 32 and count < 34:
                d += 1
            elif count >= 30 and count < 32:
                e += 1
            elif count >= 28 and count < 30:
                f += 1
            elif count >= 26 and count < 28:
                g += 1
            elif count >= 24 and count < 26:
                h += 1
            else:
                error += 1
            total += 1
        df = pd.DataFrame([[a, b, c, d, e, f, g, h, error, total]])
        df.to_csv(self.address, mode="a", header=False, index=False, sep="\t")

class scraping: # 
    def __init__(self):
        self.port = detectSerial().get_serial_list()

    def process(self):

        # Read Serial
        serialRead = ReadLine(serial.Serial(self.port, 115200))

        # Informatiom
        rounds = Prompt.ask("[bold cyan]Enter rounds[/bold cyan]")
        case = Prompt.ask("[bold cyan]Enter case[/bold cyan]")
        light = Prompt.ask(
            "[bold cyan]Enter status of light (normal/noise)[/bold cyan]")
        subject = Prompt.ask("[bold cyan]Enter name subject[/bold cyan]")
        shirt = Prompt.ask(
            "[bold cyan]Enter tone color of shirt[/bold cyan]")
        shape = Prompt.ask("[bold cyan]Enter shape of subject[/bold cyan]")
        temp = Prompt.ask("[bold cyan]Enter temperature now[/bold cyan]")
        pause = Prompt.ask(
            "[bold cyan]Enter pause time (SEC.)[/bold cyan]")
        console.print("label 0 : Normal")
        console.print("label 1 : Loss of Balance on Bed")
        console.print("label 2 : Loss of Balance on Wall")
        console.print("label 3 : Loss of Balance on Table")
        label = int(Prompt.ask("[bold cyan]Enter label[/bold cyan] "))
        number = int(Prompt.ask(
            f"[bold cyan]Enter number of frames[/bold cyan] [bold green][DEFAULT[/bold green] [bold red]60[/bold red] [bold green]FRAME][/bold green] ", default=60))
        # Will Start
        for r in range(int(rounds)):
            count_while = 0
            today = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            name = f'case-{case}-{light}light-{subject}-{shirt}-{shape}-temp{temp}'
            name = today + '-' + name
            if name in os.listdir("data"):
                console.print("File already exists.", style="bold red")
                break
            else:
                os.mkdir("./data/" + name)
                os.mkdir("./data/" + name + "/images")
                df = pd.DataFrame([["38-39", "36-37", "34-35", "32-33",
                                    "30-31", "28-29", "26-27", "24-25", "Error", "Total"]])
                df.to_csv("./data/" + name + "/count.csv", mode="a",
                          header=False, index=False, sep="\t")
            csvWrite(data=None, address="./data/" + name +
                     "/raw.csv", label=label).writeHead()
            print(f"\nRound {int(r)+1}")
            play(willstart)
            time.sleep(5)
            roundLoop = 1
            play(countdown)
            time.sleep(3)
            with console.status("[bold cyan]Scraping on tasks...", spinner="bouncingBar") as status:
                while count_while < number:
                    if (count_while == number-25):
                        play(tenframeleft)
                    try:
                        data = str(serialRead.readline())
                        data = data.replace("bytearray(b'", "")
                        data = data.replace("[", "")
                        data = data.replace("]\\r\\n')", "")
                        # print(data)
                        # print(len(data.split(',')))
                        timeNow = datetime.now()
                        if (len(data.split(',')) != 768):
                            continue
                        try:
                            plot(data, timeNow, name, roundLoop).processPlot()
                            # csvWrite(data, "./data/" + name +
                            #          "/raw.csv", label).processRaw()
                            roundLoop += 1
                            csvWrite(data, address="./data/" + name +
                                     "/raw.csv", label=label).processRaw()
                            csvWrite(data, address="./data/" + name +
                                     "/count.csv", label=label).processCount()
                            console.log("Frame [green]" + str(count_while + 1) + "[/green] of [bold green]" + str(
                                number) + "[/bold green] processed.")
                            count_while += 1
                        except Exception as e:
                            data = ''
                            console.print(e, style="bold red")
                            continue
                    except Exception as e:
                        data = ''
                        console.print(e, style="bold red")
                        continue
            today = None
            data = None
            play(rest)
            print('Break')
            if(r == int(rounds)-1):
                play(finish)
            time.sleep(int(pause))
            play(signal)

if __name__ == '__main__':
    try:
        scraping().process()
    except Exception as e:
        console.print(f"Error : {e}", style="bold red")
        scraping().process()
