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
from win10toast import ToastNotifier
from pygame import mixer
from rich.console import Console
from colorama import Fore, Back, Style
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn

console = Console()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

width = 32
height = 24

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Conv Layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.batch1 = torch.nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()

        # Max Pooling Layer 1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()

        # Max Pooling Layer 2
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, i):

        x = i.view(-1, i.shape[2], i.shape[3], i.shape[4])

        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        # Max Pool 1
        out = self.pool1(out)

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)

        out = out.view(i.shape[0], i.shape[1], -1)
        return out

# Customize the model LSTM

input_size = 9*7*32 # for 2 layers CNN
# input_size = 5*4*64 # for 3 layers CNN
# input_size = 3*3*128 # for 4 layers CNN
# input_size = 2*2*256 # for 5 layers CNN

hidden_size = 100 # Hidden Unit
layer_size = 1 # Hidden Layers

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True)

        # Output Size = Number of classes = 3
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return out

class detectSerial:
    def __init__(self):
        self.serialList = []

    def get_serial_list(self):
        for port, desc, hwid in sorted(serial.tools.list_ports.comports()):
            self.serialList.append(port)
        if len(self.serialList) > 1:
            print("Serial ports found:")
            for i in range(len(self.serialList)):
                print(Style.BRIGHT + Fore.CYAN +
                      str(i + 1) + ": " + self.serialList[i])
            return input(Style.BRIGHT + Fore.GREEN + "Select serial port :" + Style.RESET_ALL).lower()
        elif len(self.serialList) == 1:
            return self.serialList[0]
        else:
            print("No serial ports found.")
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


def transformSequenceData(data):
    X = []

    video_frame = []

    for d in (data):
        d = str(d).strip()
        d = float(d)
        frame2D = []
        frame = d
        for h in range(height):
            frame2D.append([])
            for w in range(width):
                t = frame[h * width + w]
                frame2D[h].append(t)
        video_frame.append([frame2D])
        X.append(video_frame)

    X = torch.FloatTensor(X)
    return X

# Model #
cnn_model = CNN()
lstm_model = LSTM()

cnn_model.load_state_dict(torch.load('D:\Code\CNN-LSTM\ExampleCNN-LSTM-By-Buk\model\cnn_model.pth'))
lstm_model.load_state_dict(torch.load('D:\Code\CNN-LSTM\ExampleCNN-LSTM-By-Buk\model\lstm_model.pth'))

cnn_model.to(device)
lstm_model.to(device)

cnn_model.eval()
lstm_model.eval()
##

port = detectSerial().get_serial_list()
serialRead = ReadLine(serial.Serial(port, 115200))
data_storage = []
predicted = 0

while True:
    with console.status("[bold cyan]Reading...", spinner="bouncingBar") as status:
        data = str(serialRead.readline())
        data = data.replace("bytearray(b'", "")
        data = data.replace("[", "")
        data = data.replace("]\\r\\n')", "")
        data = data.split(',')
        # print(data)
        if (len(data) != 768):
            continue
        data_storage.append(data)
        if (len(data_storage) == 5):

            inputs = transformSequenceData(data_storage)
            features = cnn_model(inputs)
            outputs = lstm_model(features)
            _, predict = torch.max(outputs.data, 1)
            predicted = predict.item()
            today = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            result = ''
            if (predicted == 0):
                result = Fore.GREEN+Style.BRIGHT+'Normal'+Fore.RESET+Style.RESET_ALL
            elif (predicted == 1):
                result = Fore.RED+Style.BRIGHT+'Loss of Balance Bed'+Fore.RESET+Style.RESET_ALL
            elif (predicted == 2):
                result = Fore.RED+Style.BRIGHT+'Loss of Balance Wall'+Fore.RESET+Style.RESET_ALL
            else:
                result = Fore.RED+Style.BRIGHT+'Normal'+Fore.RESET+Style.RESET_ALL
            print(f'[{Fore.GREEN}+{Fore.RESET}] {Fore.CYAN}Data at {Fore.GREEN}{Style.BRIGHT}{today}{Style.RESET_ALL}{Fore.RESET} Predicted :{result}{Fore.RESET}{Style.RESET_ALL}')
            data_storage = []
