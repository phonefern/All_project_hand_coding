import pandas as pd

file_path = r'C:\read_thermal\sumdata_2024\data_summary\Data_Normal_all_perfect.csv'


df = pd.read_csv(file_path)



# Assuming df is your DataFrame
label_0_data = df[df['Label'] == 0]  # Select rows where Label is 0
label_1_data = df[df['Label'] == 1]  # Select rows where Label is 1

# Combine data from both labels
combine_df = pd.concat([label_0_data, label_1_data])

# Calculate the middle index for each label
middle_index_0 = len(label_0_data) // 2
middle_index_1 = len(label_1_data) // 2

# Select the middle portion (5000 rows) for each label
middle_data_0 = label_0_data.iloc[middle_index_0 - 2500: middle_index_0 + 2500]
middle_data_1 = label_1_data.iloc[middle_index_1 - 2500: middle_index_1 + 2500]

# Combine the middle portion of both labels
middle_combined_df = pd.concat([middle_data_0, middle_data_1])

# Define the output CSV file path
output_df_file = r'C:\read_thermal\sumdata_2024\data_summary\Data_Normal_all_perfect_10000.csv'

# Save the combined DataFrame to a CSV file
middle_combined_df.to_csv(output_df_file, index=False)
