import pandas as pd

# อ่านไฟล์ CSV "Train_PP_JJ_Jakun.csv"
df_train = pd.read_csv('C:/read_thermal/Train_PP_JJ_Jakun_add.csv')

# อ่านไฟล์ CSV "9/10/2023_MeetingRoom_Data.csv"
df_additional = pd.read_csv('C:/read_thermal/SumAllCase/9_10_2023_MeetingRoom_Data.csv')

# ลบ Timestamp ปัจจุบันและเปลี่ยนเป็นชั่วโมงและนาที 4 ตัวอักษรสุดท้าย
df_additional['TimeStamp'] = df_additional['Timestamp'].apply(lambda x: x.split()[-1][:5])

# สร้างรายการเพื่อเก็บข้อมูลตัวแรกของคอลัมต่าง ๆ ซ้ำ 900 ครั้ง
frames_to_add = []

# นับว่าครบกี่ครั้งแล้ว
count = 0

# วนลูปเพื่อเลือกข้อมูลตัวแรกในคอลัมต่าง ๆ ซ้ำ 900 ครั้ง
for i in range(len(df_additional)):
    selected_frames = df_additional.iloc[i:i+1, [20, 22, 24, 26]]
    frames_to_add.extend([selected_frames] * 900)
    count += 1
    
    if count == 900:
        count = 0
        # นับครบ 900 ครั้งแล้วเปลี่ยนไปยังข้อมูลถัดไป
        i += 1  # เพื่อเลือกข้อมูลถัดไป

# สร้างคอลัมใหม่ที่จะเพิ่มลงใน "Train_PP_JJ_Jakun_add.csv"
new_columns = pd.concat(frames_to_add, ignore_index=True)

# เพิ่มคอลัมใหม่ลงใน "Train_PP_JJ_Jakun_add.csv"
df_train = pd.concat([df_train, new_columns], axis=1)

# บันทึกผลลัพธ์ลงในไฟล์ CSV ใหม่ "Train_PP_JJ_Jakun_add_temp2.csv"
df_train.to_csv('Train_PP_JJ_Jakun_add_temp2.csv', index=False)

print('Finish')
