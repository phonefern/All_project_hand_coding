import pandas as pd

# อ่านไฟล์ CSV
# final_data = pd.read_csv("C:\\read_thermal\SumAllCase/data_Noise_all_5_11.csv")
another_data = pd.read_csv(r"C:\read_thermal\sumdata_2024\data_summary\data_simulate\Data_normal_all.csv")


# นับค่าในคอลัมน์ 'Label'
# label_counts = final_data['Label'].value_counts()
label_counts2 = another_data['Label'].value_counts()

# แสดงผลการนับ
# print(f'Label_Noise > : ')
# print(label_counts)j
# print('--------------------------------')
print(f'Label_Normal > :  ')
print(label_counts2)
print('--------------------------------')