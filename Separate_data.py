import pandas as pd

df = df = pd.read_csv(r"C:\read_thermal\SumAllCase\Data_AbNormal_all.csv")

df_label_0 = df[df['Label'] == 0]

df_label_1 = df[df['Label'] == 1]

df_label_0.to_csv(r'C:\read_thermal\SumAllCase\balanced_dataset\label_all_0.csv', index=False)
df_label_1.to_csv(r'C:\read_thermal\SumAllCase\balanced_dataset\label_all_1.csv', index=False)

print('Split label is success.')