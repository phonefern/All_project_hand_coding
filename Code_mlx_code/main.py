import serial

# กำหนดพอร์ต COM ที่ต้องการใช้
# com_port = 'COM7'

# เปิดพอร์ต Serial ด้วย PySerial
ser = serial.Serial('COM8' , 115200)  # 115200 เป็นความเร็วในการสื่อสาร (baud rate) ที่ต้องตรงกับ MicroPython

try:
    while True:
        # อ่านข้อมูลจาก MicroPython
        data = ser.readline().decode('utf-8').strip()
        print(data)
except KeyboardInterrupt:
    # หยุดโปรแกรมเมื่อกด Ctrl+C
    pass
finally:
    # ปิดการเชื่อมต่อเมื่อเสร็จสิ้น
    ser.close()
