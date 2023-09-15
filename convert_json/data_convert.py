import json
import pandas as pd
from datetime import datetime

# Load JSON data from the file
with open("testdata04.json", "r") as file:
    json_data = file.read()

data = json.loads(json_data)

raw_data = data['New_path4']

combined_data = {}

for timestamp, values in raw_data.items():
    value_list = values.split(',')
    # เวลาอยู่ที่ format ของเราด้วย
    formatted_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    formatted_timestamp = formatted_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    combined_data[timestamp] = {"timestamp": formatted_timestamp}
    for i, value in enumerate(value_list):
        key = str(i)
        combined_data[timestamp][key] = value

# print("Combined Data:")
# for timestamp, value_dict in combined_data.items():
#     print(f"Timestamp: {value_dict['timestamp']}")
#     for key, value in value_dict.items():
#         print(f"{key}: {value}")
#     print()

# Convert combined_data to JSON
combined_data_json = json.dumps(combined_data)

# Write the JSON to a file
with open("sort_test_data3.json", "w") as file:
    file.write(combined_data_json)

print("JSON file created: sort_test_data.json")

df = pd.read_json("sort_test_data3.json")
df = df.transpose()  # สลับแถวกับคอลัมน์เพื่อเป็นแนวนอน

df.to_csv("DB_jsonToCSV_testData4.csv", index=False)  # บันทึกเป็นไฟล์ CSV โดยไม่รวมดัชนี