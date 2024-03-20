# import time
# import utime
# import ntptime
# import machine
# import adafruit_mlx90640
# import network
# import ufirebase as fb
# import gc

# machine.freq(240000000)
# led = machine.Pin(2, machine.Pin.OUT)
# mlx = adafruit_mlx90640.MLX90640(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21), freq=400000))
# mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

# # WiFi
# wlan = network.WLAN(network.STA_IF)
# if not wlan.active() or not wlan.isconnected():
#     wlan.active(True)
#     wlan.connect("IOT-WU", "iot123456")
# while not wlan.isconnected():
#     time.sleep(5)
#     pass
# time.sleep(5)
# gc.collect()

# # Firebase
# fb.setURL("https://sensorinfrared-ccf01-default-rtdb.asia-southeast1.firebasedatabase.app")

# # Set time
# ntptime.settime()
# tm = utime.localtime()
# tm = tm[0:3] + (0,) + tm[3:6] + (0,)
# machine.RTC().datetime(tm)
# rtc = machine.RTC()
# utc_shift = 7
# (year, month, mday, week_of_year, hour, minute, second, milisecond) = rtc.datetime()
# rtc.init((year, month, mday, week_of_year, hour + utc_shift, minute, second, milisecond))
# gc.collect()

# while True:
#     gc.collect()
#     frame = [0] * 768
#     t = rtc.datetime()
#     now = '{:04d}-{:02d}-{:02d}_{:02d}{:02d}{:02d}'.format(t[0], t[1], t[2], t[4], t[5], t[6])
#     mlx.getFrame(frame)
#     fb.put('frame/'+now, frame, bg=0)
#     gc.collect()
#     time.sleep(1)

import time
import machine
import adafruit_mlx90640
import gc

machine.freq(240000000)
mlx = adafruit_mlx90640.MLX90640(machine.I2C(0, scl=machine.Pin(22), sda=machine.Pin(21), freq=800000))
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

while True:
    try:
        gc.collect()
        frame = [0] * 768
        mlx.getFrame(frame)
        print(frame)
        gc.collect()

    except Exception as e:
        print(e)
        machine.reset()
    # gc.disable()

    finally:
        gc.collect()
        frame.clear()
    # time.sleep(0.5)