import numpy as np
import pandas as pd

# Read the CSV file
csv_file_path = "C:\\read_thermal\sumdata_2024\data_summary\data_simulate\\Data_normal_all.csv"
df = pd.read_csv(csv_file_path)

# Define the column indices for pixel values (0-767)
pixel_columns = df.columns[1:-1]

# Function to add noise to a row
def add_noise(row):
    noise = np.random.normal(0.3941, 0.2109, len(pixel_columns))
    # print(noise)
    
    # Convert values in pixel_columns to float before rounding
    row[pixel_columns] = row[pixel_columns].astype(float).round(1) + noise
    
    return row

# Apply the function to each row
df = df.apply(add_noise, axis=1)

# Save the modified DataFrame back to a new CSV file with values formatted as 0.1f
modified_csv_path = 'C:\\read_thermal\sumdata_2024\data_summary\data_simulate\\Data_All_Test_simulate.csv'
df.to_csv(modified_csv_path, index=False, float_format='%.1f')
print('Done')
