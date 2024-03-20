import numpy as np
import pandas as pd

# Read the CSV file
csv_file_path = r"C:\read_thermal\sumdata_2024\data_summary\data_simulate\\4LabelOriginal_Data.csv"
df = pd.read_csv(csv_file_path)

# Define the column indices for pixel values (0-767)
pixel_columns = df.columns[1:-1]

# Function to subtract noise from pixel values in a row
def subtract_noise(row):
    noise = np.random.normal(1.536, 0.866, len(pixel_columns))
    # print(noise)
    
    # Convert values in pixel_columns to float before rounding
    row[pixel_columns] = row[pixel_columns].astype(float).round(1) - noise
    
    return row

# Apply the function to each row
df = df.apply(subtract_noise, axis=1)

# Save the modified DataFrame back to a new CSV file with values formatted as 0.1f
modified_csv_path = 'E:\\Jay_CNN-LSTM\\Data_Sum\\4Label\\4LabelSimulate_Noise_Data_N_1.536_0.866.csv'
df.to_csv(modified_csv_path, index=False, float_format='%.1f')
