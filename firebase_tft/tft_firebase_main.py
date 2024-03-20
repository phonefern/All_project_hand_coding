#เรียกใช้ไรเบอรี่ที่ใช้งานร่วมตัวES32
import machine
import time
import adafruit_amg8833
import adafruit_st7735r
from adafruit_st7735r import TFT, TFTColor
from machine import I2C, Pin, SPI, PWM
import gc
from random import randint
import ufirebase
import ntptime
import network, socket

machine.freq(240000000) #เรียกใช้ฟังก์ชันในโมดูล 'machine' เพื่อกำหนดความถี่ของตัวประมวลในบอร์ดความเร็วบอร์ด
amg = adafruit_amg8833.AMG88XX(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21),freq=400000)) #กำหนดค่าพารามิเตอร์เพื่อเชื่อมต่อกับAMG8833 โดยมีความถี่ 400000ในการสือสาร


# Hardware SPI, HSPI กำหนดค่า และสร้าง Obj ชื่อ SPI ใน ESP32
spi = SPI(1, baudrate=8000000, polarity=0, phase=0)

# tft = TFT(spi, 16) #สร้างObj ชื่อTFT สี
# tft.init_7735(tft.BLACKTAB) #กำหนดธีมสีเป็นสีดำ

# tft.fill(TFTColor(0x02, 0x02, 0xC4)) #กำหนดสีทั้งหน้าจอเป็นสีที่กำหนด

# tft = adafruit_st7735r.TFT(machine.SPI(2, baudrate=20000000, polarity=0, phase=0, sck=machine.Pin(33), mosi=machine.Pin(32)), 16, 17, 18) #เป็นการกำหนดค่าและสร้างอ็อบเจ็กต์ TFT สำหรับการใช้งานหน้าจอ TFT รุ่น ST7735 ใน ESP32
# tft.init_7735(adafruit_st7735r.TFT.BLACK) #เริ่มต้นการใช้งาน TFT รุ่น ST7735 โดยกำหนดธีมสีดำ
# tft.fill(adafruit_st7735r.TFT.RED) #เติมสีให้กับหน้าจอ TFT ด้วยสีแดง

sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect('IOT-WIFI')
while not sta_if.isconnected():
    print('connecting to network...')
    pass

#URL of your Firebase Realtime Database
ufirebase.setURL("https://handdetect-47529-default-rtdb.firebaseio.com/")

#This queries the time from an NTP server
ntptime.settime()

##Get current time
UTC_OFFSET = +7 * 60 * 60  # '+7' is the Timezone for Thailand
actual_time = time.localtime(time.time() + UTC_OFFSET)

# Set time
rtc = machine.RTC()
rtc.datetime((actual_time[0], actual_time[1], actual_time[2], 0, actual_time[3], actual_time[4], actual_time[5], 0))

gc.collect() #เรียกใช่Garbage Collector .โมดูลที่ใช้ในการจัดการหน่วยความจำแบบอัตโนมัติ.ทำการล้างและตรวจสอบพร้อมกับลบ Obj ที่ไม่ถูกใช้

while True: #ใช้คำสั่งนี้ในการวนลูปเมื่อมีไฟเลี้ยงบอร์ด
    #กำหนดค่าตัวแปรความกว้างสูงของขนาดของสี่เหลี่ยมที่ใช้ในการแสดงผลพิกเซลของเซ็นเซอร์อุณหภูมิ AMG8833 บนหน้าจอ TFT
    gc.enable()
    # rect_hig = 20
    # rect_wid = 16
   
    try:
        
        amg.refresh()
        amgpixel = [amg[row, col] for row in range(8) for col in range(8)]
        print(amgpixel) #แสดงค่าตัวแปร amgpixel

        t = rtc.datetime()
        currentTime = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(t[0], t[1], t[2], t[4], t[5], t[6])

        message = ",".join(str(x) for x in amgpixel)
        
        # for h in range(8): #วนลูปตามตัวแปร h ซึ่งจะเป็นตัวเลขตั้งแต่ 0 ถึง 7 เพื่อแทนแถว (row) ของพิกเซลในตาราง 8x8
        #     for w in range(8): #วนลูปตามตัวแปร w ซึ่งจะเป็นตัวเลขตั้งแต่ 0 ถึง 7 เพื่อแทนคอลัมน์ (column) ของพิกเซลในตาราง 8x8
        #         t = amgpixel[h * 8 + w] #กำหนดค่า t โดยใช้ค่าพิกเซลที่ตำแหน่ง h, w ขนาด 8*8

        #         color = int((t - 20) / 20 * 255) #การคำนวณหาค่าสีที่ใช้แสดงผล
        #         TFTCOLOR = TFTColor(color, color, color) #ตัวแปรสีที่ใช้กำหนดค่าสีที่แสดงผล

        #         if t < 25:
        #             TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red
        #         elif t < 26:
        #             TFTCOLOR = TFTColor(0xFF, 0x80, 0x00)  # Orange
        #         elif t < 27:
        #             TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow
        #         elif t < 28:
        #             TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  # Green
        #         elif t < 29:
        #             TFTCOLOR = TFTColor(0x00, 0xFF, 0xFF)  # Cyan
        #         elif t < 30:
        #             TFTCOLOR = TFTColor(0x00, 0x80, 0xFF)  # Light Blue
        #         elif t < 31:
        #             TFTCOLOR = TFTColor(0x00, 0x00, 0xFF)  # Blue
        #         elif t < 32:
        #             TFTCOLOR = TFTColor(0x80, 0x00, 0xFF)  # Purple
        #         elif t < 33:
        #             TFTCOLOR = TFTColor(0xFF, 0x00, 0xFF)  # Magenta
        #         elif t < 34:
        #             TFTCOLOR = TFTColor(0xFF, 0x00, 0x80)  # Pink
        #         elif t < 35:
        #             TFTCOLOR = TFTColor(0xFF, 0x00, 0x40)  # Dark Red
        #         elif t < 36:
        #             TFTCOLOR = TFTColor(0xFF, 0x40, 0x40)  # Dark Orange
        #         elif t < 37:
        #             TFTCOLOR = TFTColor(0xFF, 0x80, 0x80)  # Light Red
        #         elif t < 38:
        #             TFTCOLOR = TFTColor(0x80, 0xFF, 0x80)  # Light Green
        #         elif t < 39:
        #             TFTCOLOR = TFTColor(0x40, 0xFF, 0xFF)  # Light Cyan
        #         elif t >= 39:
        #             TFTCOLOR = TFTColor(0x00, 0x00, 0xFF)  # Blue

                # if (t < 22.00):TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  # Green color
                # elif(t < 23.00): TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  #(สีน้ำตาลอ่อน
                # elif(t < 24.00): TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  #(สีเหลืองอ่อน)
                # elif(t < 25.00): TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  #(สีเหลืองอ่อน)
                # elif(t < 26.00): TFTCOLOR = TFTColor(0x00, 0xFF, 0x00)  #(สีเหลืองอ่อน)
                # elif(t < 27.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 28.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 29.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 30.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 31.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 32.00): TFTCOLOR = TFTColor(0xFF, 0xFF, 0x00)  # Yellow color
                # elif(t < 33.00): TFTCOLOR = TFTColor(0xFF, 0xA5, 0x00)  # Orange color
                # elif(t < 34.00): TFTCOLOR = TFTColor(0xFF, 0xA5, 0x00)  # Orange color
                # elif(t < 35.00): TFTCOLOR = TFTColor(0xFF, 0xA5, 0x00)  # Orange color
                # elif(t < 36.00): TFTCOLOR = TTFTColor(0xFF, 0xA5, 0x00)  # Orange color
                # elif(t < 37.00): TFTCOLOR = TFTColor(0xFF, 0xA5, 0x00)  # Orange color
                # elif(t > 38.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 39.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 40.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 41.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 42.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 43.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 44.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color
                # elif(t > 45.00):  TFTCOLOR = TFTColor(0xFF, 0x00, 0x00)  # Red color



                # tft.fillrect((w * rect_wid, h * rect_hig), (rect_wid, rect_hig), TFTCOLOR) #

        try:
            path = "Hand01/" + currentTime + "/"
            ufirebase.put(path, message, bg=0)
            print("Data sent successfully!\nAt Time:", currentTime)
            print("-" * 50)
        except Exception as e:
            print('Cannot Publish to Google Cloud...', e)
            machine.reset()


        gc.collect()

    except Exception as e: #เก็บข้อมูลข้อผิดพลาดและดำเนินการตามที่ต้องการ
        print(e) #แสดงข้อมูลe
        machine.reset() #ทำการรีเซ็ตใน
    gc.disable()