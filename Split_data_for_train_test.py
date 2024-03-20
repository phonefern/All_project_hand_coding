import pandas as pd

file_path = r'C:\read_thermal\sumdata_2024\data_summary\Normal_split_30000.csv'

df = pd.read_csv(file_path)

# Filter rows with 'Label' equal to 0
Label_0 = df[df['Label'] == 0]

# Split Label_0 into two dataframes with a 70-30 ratio
split_index = int(0.7 * len(Label_0))
Label_0_70 = Label_0.head(split_index)
Label_0_30 = Label_0.tail(len(Label_0) - split_index)

# Output file paths
output_df_file_70 = r'C:\read_thermal\sumdata_2024\data_summary\Label_0_70_normal.csv'
output_df_file_30 = r'C:\read_thermal\sumdata_2024\data_summary\Label_0_30_normal.csv'

# Export dataframes to CSV files
Label_0_70.to_csv(output_df_file_70, index=False)
Label_0_30.to_csv(output_df_file_30, index=False)


# Filter rows with 'Label' equal to 0
Label_1 = df[df['Label'] == 1]

# Split Label_0 into two dataframes with a 70-30 ratio
split_index = int(0.7 * len(Label_1))
Label_1_70 = Label_1.head(split_index)
Label_1_30 = Label_1.tail(len(Label_1) - split_index)

# Output file paths
output_df_file_70_1 = r'C:\read_thermal\sumdata_2024\data_summary\Label_1_70_normal.csv'
output_df_file_30_1 = r'C:\read_thermal\sumdata_2024\data_summary\Label_1_30_normal.csv'

# Export dataframes to CSV files
Label_1_70.to_csv(output_df_file_70_1, index=False)
Label_1_30.to_csv(output_df_file_30_1, index=False)
