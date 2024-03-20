import csv
import statistics
import pandas as pd
# อ่านไฟล์ CSV

df = pd.read_csv(r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\Data_normal_all.csv')

# df.drop('TimeStamp', axis=1, inplace=True)

# เก็บคอลัมน์ 'Label' เพื่อใช้งานในภายหลัง
label_column = df['Label']

# ลบคอลัมน์ 'Label' ออกจาก DataFrame
df = df.drop('Label', axis=1)


# คำนวณค่า Max, Mean, และ Min ในแต่ละแถว
df['Max'] = df.iloc[:, :].max(axis=1, numeric_only=True)
df['Mean'] = df.iloc[:, :].mean(axis=1, numeric_only=True)
df['Min'] = df.iloc[:, :].min(axis=1, numeric_only=True)
df['SD'] = df.iloc[:, :].std(axis=1, numeric_only=True)

# เรียกคอลัมน์ 'Label' กลับมาและเพิ่มไปอยู่หลังคอลัมน์ 'Min'
df['Label'] = label_column

# บันทึกผลลัพธ์ลงในไฟล์ CSV ใหม่
df.to_csv(r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\data_simulate_Normal_01.csv', index=False)
