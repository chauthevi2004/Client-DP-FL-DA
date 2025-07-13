import pandas as pd

LABELS = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','healthy']
SPLITS = ['train', 'validate', 'test']

for split in SPLITS:
    print(f"\n--- {split.upper()} ---")
    csv_path = f"../dataset/RSNA-ICH/binary_25k/{split}.csv"
    df = pd.read_csv(csv_path)
    total = len(df)
    for label in LABELS:
        count = df[label].sum()
        ratio = count / total
        print(f"{label:18}: {int(count):6} ({ratio:.2%})")
    print(f"Total samples: {total}") 