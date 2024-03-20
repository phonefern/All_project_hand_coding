import os
import pandas as pd

# กำหนดโฟลเดอร์หลักที่มีโฟลเดอร์ย่อยที่เก็บไฟล์ CSV
main_folder = r"C:\read_thermal\data_v2"

# รายการเพื่อเก็บผลลัพธ์
results = []

# วนลูปผ่านโฟลเดอร์ในโฟลเดอร์หลัก
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)

    # ตรวจสอบว่าเป็นโฟลเดอร์หรือไม่
    if os.path.isdir(subfolder_path):
        # สร้างรายการเพื่อเก็บข้อมูลจากไฟล์ CSV
        folder_data = []

        # วนลูปผ่านไฟล์ CSV ในโฟลเดอร์ย่อย
        for csv_file in os.listdir(subfolder_path):
            if csv_file.endswith(".csv"):
                csv_file_path = os.path.join(subfolder_path, csv_file)

                # อ่านข้อมูลจากไฟล์ CSV และรวมเข้ากับรายการ folder_data
                df = pd.read_csv(csv_file_path)
                folder_data.append(df)

        # รวมข้อมูลจากทุกไฟล์ CSV ในโฟลเดอร์ย่อย
        if folder_data:
            combined_data = pd.concat(folder_data, ignore_index=True)
            results.append(combined_data)

# รวมข้อมูลจากทุกโฟลเดอร์ย่อย
if results:
    final_data = pd.concat(results, ignore_index=True)
    print(final_data)
# บันทึกผลลัพธ์ในไฟล์ CSV
    output_file = r"C:\read_thermal\sumdata_2024\data_summary/Data_AbNormal_all.csv"
    final_data.to_csv(output_file, index=False)