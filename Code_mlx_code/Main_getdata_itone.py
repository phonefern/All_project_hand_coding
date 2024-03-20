import time
import serial.tools.list_ports
import datetime
import pandas as pd
import os
import pygame
from rich.console import Console
from rich.prompt import Prompt

def switch_scenarios(inputscenarios):
    switch = {
        '0': "0-Laydown-Scenario",
        '1': "1-Blacket-Scenario",
        '2': "2-No-Blacket-Scenario",
        '3': "3-Blacket-With-Fan-Scenario",
        '4': "4-No-Blacket-With-Fan-Scenario"
    }
    return switch.get(inputscenarios, "Invalid Scenario")


def switch_case(inputcase):
    switch = {
        '0': "0-Underweight",
        '1': "1-Helthy-Weight",
        '2': "2-Overweight",
        '3': "3-Obesity"

    }
    return switch.get(inputcase, "Invalid Case")

def switch_Post(inputpost):
    switch = {
        '0': "Laydown",
        '1': "Brachial-90",
        '2': "Brachial-45",
        '3': "Shoulder-90",
        '4': "Shoulder-45",
    }
    return switch.get(inputpost, "Invalid Post")

def switch_labelcase(inputlabel):
    switch = {
        '0': "0",
        '1': "1",
    }
    return switch.get(inputlabel, "Invalid Case")


Subject = Prompt.ask("[bold red]Enter Name Subject[/bold red]")
Noise = Prompt.ask("[bold red]Enter Noise [With Noise][/bold red]")
name = Subject
noise = Noise
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portList = []

for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

val = input("select Port: COM")

for x in range(0, len(portList)):
    if portList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portList[x])

dt = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
dt_str = dt.replace(":", ".")
date = datetime.datetime.now().strftime("%d-%m-%Y")
print("Input 0: Laydown-Scenario")
print("Input 1: Blacket-Scenario")
print("Input 2: No-Blacket-Scenario")
print("Input 3: Blacket-With-Fan-Scenario)")
print("Input 4: No-Blacket-With-Fan-Scenario")
inputscenarios = input("Select Scenarios: ")

print("Input 0: Underweight")
print("Input 1: Helthy")
print("Input 2: Overweight")
print("Input 3: Obesity")
inputcase = input("Select Case: ")

print("Input 0: Laydown")
print("Input 1: Brachial-90")
print("Input 2: Brachial-45")
print("Input 3: Shoulder-90")
print("Input 4: Shoulder-45")
inputpost = input("Select Case: ")

print("Input 0: Normal")
print("Input 1: Abnormal")
inputlabel = input("Select Case: ")

textscenarios = switch_scenarios(inputscenarios)
textcase = switch_case(inputcase)
textpost = switch_Post(inputpost)
textlabelcase = switch_labelcase(inputlabel)

if textscenarios != "Invalid Scenario" and textcase != "Invalid Case":
    print(f"Selected Scenarios: {textscenarios}")
    print(f"Selected Case: {textcase}")
else:
    print("Invalid input. Please select a valid scenario and case.")

serialInst.baudrate = 115200
serialInst.port = portVar
serialInst.open()

combined_data = {}  # สร้างดิกชันนารีเพื่อเก็บข้อมูลที่รวมทั้งหมด


pygame.mixer.init()
sound1 = pygame.mixer.Sound(r"C:\HAND_GOD_GIVE\PK_Readse\1.จะเริ่มเก็บข้อมูลในอีกห้.mp3")
sound2 = pygame.mixer.Sound(r"C:\HAND_GOD_GIVE\PK_Readse\2.ขอบคุณที่อดทนเก็บข้อมูลเ.mp3")
# Play sound1 to indicate data collection has started
sound1.play()
time.sleep(10)  # Sleep for 10 seconds
count = 0
while True:
    try:
        if serialInst.in_waiting:
            data_str = serialInst.readline().decode("UTF-8").strip()
            data_str = data_str.replace('[', '').replace(']', '')  # ลบ [ และ ] ออกจากข้อมูล
            data_list = [float(x) for x in data_str.split(',')]
            cdt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Import Data into combined_data
            timestamp = cdt
            combined_data[timestamp] = {"timestamp": timestamp}
            for i, v in enumerate(data_list):
                key = str(i)
                combined_data[timestamp][key] = v
            combined_data[timestamp]["label"] = textlabelcase

            df = pd.DataFrame.from_dict(combined_data, orient='index')  # Add Data to CSV

            # ตรวจสอบว่าเส้นทาง D:\ESP32\{textcase} มีอยู่หรือไม่ ถ้าไม่มีให้สร้าง
            if not os.path.exists(f"C:\HAND_GOD_GIVE\PK_Readse\data\\{textscenarios}\\{textcase}\\{date}\\{dt_str}_{textcase}-{name}-{textpost}-Label-{textlabelcase}-{noise}\\img"):
                os.makedirs(f"C:\HAND_GOD_GIVE\PK_Readse\data\\{textscenarios}\\{textcase}\\{date}\\{dt_str}_{textcase}-{name}-{textpost}-Label-{textlabelcase}-{noise}\\img")

            # กำหนดเส้นทางแบบถูกต้องสำหรับไฟล์ CSV
            csv_file_path = f"C:\HAND_GOD_GIVE\PK_Readse\data\\{textscenarios}\\{textcase}\\{date}\\{dt_str}_{textcase}-{name}-{textpost}-Label-{textlabelcase}-{noise}\\rawData.csv"

            df.to_csv(csv_file_path, index=False, mode='w')  # สร้างไฟล์ CSV

            print("Data saved to CSV: {0}".format(count), end="\r")
            count += 1 # เพิ่มค่า count ทุกครั้งที่เพิ่มข้อมูล

            if count == 900:
                sound2.play()
                time.sleep(2)
                serialInst.close()
                break

    except KeyboardInterrupt:
        # sound2.play()
        time.sleep(2)
        serialInst.close()
        break

    except Exception as e:
        print(f"Error: {e}")

print("Get Data Success.")
