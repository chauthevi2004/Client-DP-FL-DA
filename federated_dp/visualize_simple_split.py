import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def simple_split(df, n_clients=20, random_state=42):
    """
    Chia dữ liệu ngẫu nhiên đều cho các client (IID split).
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    clients = np.array_split(df, n_clients)
    return clients

# --- Đọc dữ liệu ---
# Thay đường dẫn này bằng file train.csv thực tế của bạn
csv_path = "../dataset/RSNA-ICH/binary_25k/train.csv"
df = pd.read_csv(csv_path)

# --- Chia dữ liệu ---
LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']
clients = simple_split(df, n_clients=20)

# --- Tính phân phối nhãn ---
label_counts = []
for i, cdf in enumerate(clients):
    counts = [cdf[label].sum() for label in LABELS]
    label_counts.append(counts)
    print(f"Client {i}:")
    for label, count in zip(LABELS, counts):
        print(f"  {label}: {int(count)}")
    print(f"  Total: {int(sum(counts))}\n")
label_counts = np.array(label_counts)

# --- Vẽ biểu đồ ---
plt.figure(figsize=(18, 8))
for i in range(len(clients)):
    plt.subplot(4, 5, i+1)
    sns.barplot(x=LABELS, y=label_counts[i], palette="viridis")
    plt.title(f"Client {i}")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.suptitle("Label distribution per client", y=1.02, fontsize=18)
plt.tight_layout()
plt.savefig("simple_label_distribution_per_client.png")
plt.show()