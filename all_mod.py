import os
import pandas as pd

# ระบุเส้นทางไฟล์ CSV ทั้งหมด
file_paths = [
r'C:\read_thermal\sumdata_2024\data_summary/evaluate_20000_predict.csv', 
r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 01 _P/Data_simulate_01_P_0.39411_0.2109.csv', 
r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 01 _N/Data_simulate_01_N_0.39411_0.2109.csv', ]
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test/simulate_02_N_split20.csv', 
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test/simulate_03_P_split20.csv', 
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test/simulate_03_N_split20.csv', 
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test/simulate_04_P_split20.csv',
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test/simulate_04_N_split20.csv',  
# r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 04 _P/Data_simulate_03_N_0.39411_0.2531.csv'

# สร้างรายการเปล่าๆเพื่อเก็บข้อมูลจากไฟล์ CSV แต่ละไฟล์
frames = []

# อ่านข้อมูลจากแต่ละไฟล์และเก็บในรายการ frames
for file_path in file_paths:
    df = pd.read_csv(file_path)
    frames.append(df)

# รวมข้อมูลจาก frames ทั้งหมดเข้าด้วยกัน
result = pd.concat(frames, ignore_index=True)

# แสดงผลลัพธ์หลังจากรวม
print(result)

# บันทึกลงไฟล์ CSV หากต้องการ
result.to_csv(r'C:\read_thermal\sumdata_2024\data_summary\evaluate_20000_predict_Noise_0.csv', index=False)
