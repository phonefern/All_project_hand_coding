import machine
import gc
import utime
from machine import I2C, Pin
import adafruit_amg8833
import network
import ntptime
import ufirebase
import urequests

print("Start Program...")

# Function to Get CurrentTime from NTP Server
def get_current_time():
    actual_time = utime.localtime(utime.time() + 25200)
    rtc = machine.RTC()
    rtc.datetime((actual_time[0], actual_time[1], actual_time[2], 0, actual_time[3], actual_time[4], actual_time[5], 0))
    t = rtc.datetime()
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(t[0], t[1], t[2], t[4], t[5], t[6])


# Function Send Data to Firebase
def data_to_firebase(currentTime, message):
    try:
        path = "TEST/" + currentTime + "/"
        ufirebase.put(path, message)
        print('Send Data to Firebase Success!')
    except Exception as e:
        print(e)
        machine.reset()


# Set CPU Frequency
machine.freq(240000000)

# Adafruit MLX90640
i2c = I2C(0, scl=Pin(22, Pin.OUT), sda=Pin(21, Pin.OUT), freq=800000)
# mlx = adafruit_mlx90640.MLX90640(i2c)
# mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
machine.freq(240000000) #เรียกใช้ฟังก์ชันในโมดูล 'machine' เพื่อกำหนดความถี่ของตัวประมวลในบอร์ดความเร็วบอร์ด
amg = adafruit_amg8833.AMG88XX(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21),freq=400000))

# Setting Wi-Fi
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect('IR_Lab','ccsadmin')  
while not sta_if.isconnected():
    pass

# URL of your Firebase Realtime Database
ufirebase.setURL('https://handdetect-47529-default-rtdb.firebaseio.com/') 

while True:
    try:
        gc.collect()
        ntptime.settime()
        # frame = [0] * 768
        # mlx.getFrame(frame)
        amg.refresh()
        amgpixel = [amg[row, col] for row in range(8) for col in range(8)]
        message = ",".join(map(str, amgpixel))
        data_to_firebase(get_current_time(), message)
        # utime.sleep(5)

    except Exception as e:
        print(e)
        machine.reset()

    finally:
        gc.collect()
        # amgpixel = []





# import machine
# import gc
# import micropython
# import utime
# from machine import I2C, Pin
# import adafruit_amg8833
# import network
# import ntptime
# import urequests

# # Function to Get CurrentTime from NTP Server
# def get_current_time():
#     actual_time = utime.localtime(utime.time() + 25200)
#     rtc = machine.RTC()
#     rtc.datetime((actual_time[0], actual_time[1], actual_time[2], 0, actual_time[3], actual_time[4], actual_time[5], 0))
#     t = rtc.datetime()
#     return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(t[0], t[1], t[2], t[4], t[5], t[6])

# # Set CPU Frequency
# machine.freq(240000000)

# # Adafruit MLX90640
# i2c = I2C(0, scl=Pin(22, Pin.OUT), sda=Pin(21, Pin.OUT), freq=800000)
# amg = adafruit_amg8833.AMG88XX(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21),freq=400000))

# # Setting Wi-Fi
# sta_if = network.WLAN(network.STA_IF)
# sta_if.active(True)
# sta_if.connect('IR_Lab', 'ccsadmin')
# while not sta_if.isconnected():
#     pass

# amgpixel_data = []


# while True:
#     try:
        
#         gc.collect()
#         ntptime.settime()
#         # frame = [0] * 768
#         # mlx.getFrame(frame)
#         amg.refresh()
#         amgpixel = [amg[row, col]
#                     for row in range(8)
#                     for col in range(8)]
#         # print("-----------------------")
#         # print(len(amgpixel))
#         # print("-----------------------")
#         amgpixel_data.append(amgpixel)
#         if len(amgpixel_data) == 16:
#             average_amgpixel = [sum(x) / 16 for x in
#                                 zip(*amgpixel_data)]  # หาค่าเฉลี่ยของค่าอุณหภูมิที่วัดได้จากเซนเซอร์ โดยนำค่าที่วัดได้มาบวกกันแล้วหารด้วยจำนวนค่าที่วัดได้ 8 ค่า แล้วเก็บค่าเฉลี่ยไว้ในตัวแปร average_amgpixel
#             amgpixel_data = []  # เมื่อหาค่าเฉลี่ยของค่าอุณหภูมิที่วัดได้จากเซนเซอร์เสร็จแล้ว ให้เคลียร์ค่าที่เก็บไว้ในตัวแปร amgpixel_data ทิ้ง

#             # ปัดทศนิยมของค่าเฉลี่ยเป็น 2 ตำแหน่งหลังจากหาค่าเฉลี่ย
#             average_amgpixel_rounded = [round(x, 2) for x in
#                                         average_amgpixel]  # ปัดทศนิยมของค่าเฉลี่ยเป็น 2 ตำแหน่งหลังจากหาค่าเฉลี่ย
#             message = ",".join(str(x) for x in average_amgpixel_rounded)
#             print(message)
#             urequests.patch('handdetect-47529-firebase-adminsdk-24820-86efffd565.json',
#                         json={get_current_time(): message}).close()
#         gc.collect()

#     except Exception as e:
#         print(f"Error: {e}")
#         machine.reset()