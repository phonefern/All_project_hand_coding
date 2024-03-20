import random
import pandas as pd
import os

INPUT_DATASET = r"C:\read_thermal\SumAllCase\Alldata_normal.csv"
OUTPUT_DIR = r"C:\read_thermal\SumAllCase"
LABEL_COL = "Label"
FRAMES_PER_SPLIT = 100000

# Load the dataset
dataset = pd.read_csv(INPUT_DATASET)

# Separate the dataset into two based on the label
dataset_0 = dataset[dataset[LABEL_COL] == 0]
dataset_1 = dataset[dataset[LABEL_COL] == 1]

# Figure out the minimum number of samples for both labels
min_num_rows = min(dataset_0.shape[0], dataset_1.shape[0])
print(f"There were {dataset_0.shape[0]} samples with label 0 and {dataset_1.shape[0]} samples with label 1. The kept amount is {min_num_rows}.")

# Randomly select the minimum number of rows for both 0s and 1s
chosen_ids_0 = random.sample(list(dataset_0.index), min_num_rows)
chosen_ids_1 = random.sample(list(dataset_1.index), min_num_rows)

# Combine the chosen indices for both labels
chosen_ids = chosen_ids_0 + chosen_ids_1

# Use the chosen indices to create the balanced dataset
balanced_dataset = dataset.loc[chosen_ids]

# Shuffle the balanced dataset
balanced_dataset = balanced_dataset.sample(frac=1).reset_index(drop=True)

# Split the balanced dataset into chunks of 10,000 frames
split_datasets = [balanced_dataset.iloc[i:i + FRAMES_PER_SPLIT] for i in range(0, len(balanced_dataset), FRAMES_PER_SPLIT)]

# Save each split
for i, split_dataset in enumerate(split_datasets):
    output_split_path = os.path.join(OUTPUT_DIR, f"split_normal_100k_{i + 1}.csv")
    split_dataset.to_csv(output_split_path, index=False)
    print(f"Saved {output_split_path}.")
    print(f"Split {i + 1} has {split_dataset.shape[0]} rows.")
