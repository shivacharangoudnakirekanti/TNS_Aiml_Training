import pandas as pd
from pathlib import Path

# Find CSV files inside the dataset folder
csv_files = list(Path("dataset").glob("*.csv"))

if not csv_files:
    print("No CSV file found inside the dataset folder.")
else:
    for file in csv_files:
        print(f"\nDataset: {file}")
        df = pd.read_csv(file)

        print("\nShape (rows, columns):")
        print(df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head().to_string())

        print("\nData types:")
        print(df.dtypes)

        print("\nMissing values:")
        print(df.isnull().sum())