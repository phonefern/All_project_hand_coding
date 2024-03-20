import numpy as np
import pandas as pd
from tqdm import tqdm

# Read the CSV file
csv_file_path = (r'C:\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 02 _N\Training_2024_prepare_simu.csv')
df = pd.read_csv(csv_file_path)

# Define the column indices for pixel values (0-767)
pixel_columns = df.columns[1:-1]

#Test P1_N
# means = [0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411]
# sds = [0.2109, 0.0422, 0.0844, 0.1265, 0.1687, 0.2109, 0.2531]

#Test P2_N
means = [0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411]
sds = [0.2109, 0.0109, 0.1891, 0.3891, 0.5891, 0.7891, 0.9891]

#Test P4_N
# means = [0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411, 0.39411]
# sds = [0.2109, 1.0545, 0.5273, 0.3515, 0.2636, 0.2109, 0.1758]

#Test P2_N
# means = [0.3941, 0.3941, 0.3941, 0.3941, 0.3941, 0.3941, 0.3941, 0.3941, 0.3941,0.3941, 0.3941]
# sds = [1.5563, 0.0778, 0.1556, 0.2334, 0.3113, 0.3891, 0.4669, 0.5447, 0.6225, 0.7003, 0.7782]



# Iterate over each combination of mean and sd
for mean, sd in zip(means, sds):
    # Function to add noise to a row
    def subtract_noise(row):
        noise = np.random.normal(mean, sd, len(pixel_columns))

        # Convert values in pixel_columns to float before rounding
        row[pixel_columns] = row[pixel_columns].astype(float).round(1) - noise

        return row


    # Apply the function to each row with a progress bar
    tqdm.pandas(desc=f"Adding Noise (mean={mean}, sd={sd})", position=0, leave=True)
    df_modified = df.progress_apply(subtract_noise, axis=1)

    # Save the modified DataFrame to a new CSV file with values formatted as 0.1f
    modified_csv_path = f'C:\\read_thermal\sumdata_2024\data_summary\data_simulate\simulate 02 _N\\Data_simulate_02_N_{mean}_{sd}.csv'
    df_modified.to_csv(modified_csv_path, index=False, float_format='%.1f')