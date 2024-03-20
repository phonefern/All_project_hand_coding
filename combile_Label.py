import pandas as pd

file_path = r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 04 _N\Data_simulate_04_N_0.39411_1.0545.csv'


df = pd.read_csv(file_path)

Label_0 = df[df['Label'] == 0].tail(12000)
Label_1 = df[df['Label'] == 1].tail(12000)

combine_df = pd.concat([Label_0, Label_1])

output_df_file = r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test\simulate_04_N_split20.csv'

combine_df.to_csv(output_df_file, index=False)

