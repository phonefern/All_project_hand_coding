import time
import machine
import adafruit_amg8833
import gc

machine.freq(240000000) #เรียกใช้ฟังก์ชันในโมดูล 'machine' เพื่อกำหนดความถี่ของตัวประมวลในบอร์ดความเร็วบอร์ด
amg = adafruit_amg8833.AMG88XX(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21),freq=400000)) #กำหนดค่าพารามิเตอร์เพื่อเชื่อมต่อกับAMG8833 โดยมีความถี่ 400000ในการสือสาร
# mlx = adafruit_mlx90640.MLX90640(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21),freq=400000)) #กำหนดค่าพารามิเตอร์เพื่อเชื่อมต่อกับMLX90640 โดยมีความถี่ 400000ในการสือสาร
# mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ #กำหนดค่าพารามิเตอร์ในการรีเฟรชค่าใน MLX90640 โดยมีค่า 8HZ

amg.hi_res(False) #กำหนดค่าพารามิเตอร์ในการรีเฟรชค่าใน AMG8833 โดยมีค่า True 
time.sleep(1)
print('------------------------------------------------------------------------------------------')
amgpixel_data = []

while True:
    try:
        gc.collect()
        amg.refresh()
        amgpixel = [amg[row, col] 
                    for row in range(8) 
                    for col in range(8)]
        # print("-----------------------")
        # print(len(amgpixel))
        # print("-----------------------")
        amgpixel_data.append(amgpixel)
        if len(amgpixel_data) == 16:
            average_amgpixel = [sum(x) / 16 for x in zip(*amgpixel_data)] #หาค่าเฉลี่ยของค่าอุณหภูมิที่วัดได้จากเซนเซอร์ โดยนำค่าที่วัดได้มาบวกกันแล้วหารด้วยจำนวนค่าที่วัดได้ 8 ค่า แล้วเก็บค่าเฉลี่ยไว้ในตัวแปร average_amgpixel
            amgpixel_data = [] #เมื่อหาค่าเฉลี่ยของค่าอุณหภูมิที่วัดได้จากเซนเซอร์เสร็จแล้ว ให้เคลียร์ค่าที่เก็บไว้ในตัวแปร amgpixel_data ทิ้ง
            
            # ปัดทศนิยมของค่าเฉลี่ยเป็น 2 ตำแหน่งหลังจากหาค่าเฉลี่ย
            average_amgpixel_rounded = [round(x, 2) for x in average_amgpixel] #ปัดทศนิยมของค่าเฉลี่ยเป็น 2 ตำแหน่งหลังจากหาค่าเฉลี่ย 
            message = ",".join(str(x) for x in average_amgpixel_rounded)

            # print(message)
            print(average_amgpixel_rounded)
            
        gc.collect()

    except Exception as e:
        print(e)
        machine.reset()
    gc.disable()


