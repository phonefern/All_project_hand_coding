import pandas as pd

# โหลดข้อมูล CSV
df = pd.read_csv(r'C:\read_thermal\sumdata_2024\data_summary\evaluate_20000_predict.csv')

# ทำการคัดลอกและวางซ้ำ 5 ครั้ง
df_copied = pd.concat([df] * 3, ignore_index=True)

# บันทึก DataFrame ที่ได้ลงในไฟล์ CSV ใหม่
df_copied.to_csv(r'C:\read_thermal\sumdata_2024\data_summary\evaluate_20000_predict_mod3.csv', index=False)
