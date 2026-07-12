import pandas as pd
df = pd.read_csv("../bank_transactions_data_2_augmented_clean_2.csv")
print(df.head())
print(df.columns)
print(df.describe())
