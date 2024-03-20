import pandas as pd

input_data_1 = r'C:\read_thermal\sumdata_2024\data_summary\evaluate_20000_predict.csv'
input_data_2 = r'C:\read_thermal\sumdata_2024\data_summary\Data_simulate_01_N_0.39411_0.4109.csv'


#combine data


df1 = pd.read_csv(input_data_1)
df2 = pd.read_csv(input_data_2)

combine_data = pd.concat([df1,df2], ignore_index=True) # 

output_data = r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\predict_test.csv'

combine_data.to_csv(output_data, index=False)

print("Combined data saved to:", output_data)