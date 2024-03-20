import os
import pandas as pd

# Define the directory where your files are located
directory = r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\split_20_test'

# Initialize an empty list to store DataFrames
data_frames = []

# Loop through each file in the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):  # Make sure to process only CSV files
            file_path = os.path.join(root, file)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Process the data (assuming similar processing as before)
            Label_0 = df[df['Label'] == 0].tail(12000)
            Label_1 = df[df['Label'] == 1].tail(12000)
            combine_df = pd.concat([Label_0, Label_1])
            
            # Append the processed DataFrame to the list
            data_frames.append(combine_df)

# Concatenate all DataFrames in the list into one DataFrame
combined_data = pd.concat(data_frames)

# Define the output file path
output_file_path = r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\combined_data_split_20_simu_test.csv'

# Save the combined data to a CSV file
combined_data.to_csv(output_file_path, index=False)

