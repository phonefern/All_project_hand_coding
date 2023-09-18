import serial
import time

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

try:
    readline = ReadLine(ser)    # สร้างตัวเเปร readline โดยให้เป็น class ReadLine
    while True:  #
        line = readline.readline().decode('utf-8')  # อ่านและแปลงข้อมูลเป็น string
        print(line.strip())  # แสดงข้อมูลและลบช่องว่างและตัวขึ้นบรรทัดใหม่ที่เกิดขึ้น
except KeyboardInterrupt:
    pass

# while True: 
#     serial_read = ser.readline().decode()  #อ่านข้อมูลจาก serial port ที่เชื่อมต่อกับ Arduino esp32 decode เเปลงข้อมูลเป็น string
#     print(serial_read)

# ser.close()
