import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def split_label_majority(df, labels, n_clients=20, random_state=42):
    """
    Chia đều sample cho các client, mỗi client ưu tiên 1 nhãn chính (label-majority).
    Đã chỉnh sửa để phân bổ samples của nhãn ưu tiên đều đặn trong nhóm client cùng ưu tiên một nhãn, tránh tình trạng client sau thiếu samples.
    Thêm oversampling (with replacement) nếu không đủ samples unique để đạt tỷ lệ 70%.
    """
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    client_size = len(df) // n_clients
    extra = len(df) % n_clients
    client_sizes = [client_size + (1 if i < extra else 0) for i in range(n_clients)]
    
    # Định nghĩa nhóm client theo label major
    groups = {
        'healthy': list(range(0, 4)),
        'subdural': list(range(4, 8)),
        'epidural': list(range(8, 12)),
        'intraparenchymal': list(range(12, 16)),
        'subarachnoid': [16, 18],
        'intraventricular': [17, 19]
    }
    
    # Pre-allocate major samples cho từng client
    major_allocations = [pd.DataFrame() for _ in range(n_clients)]
    used_idx = set()
    
    for label, client_list in groups.items():
        if not client_list:
            continue
        label_samples = df[df[label] == 1].sample(frac=1, random_state=random_state)  # Shuffle samples
        num_clients_in_group = len(client_list)
        available = len(label_samples)
        if available == 0:
            continue
        
        # Phân bổ đều samples unique available cho các client trong nhóm
        base_per_client = available // num_clients_in_group
        extra_per_client = available % num_clients_in_group
        label_client_alloc_sizes = [base_per_client + (1 if i < extra_per_client else 0) for i in range(num_clients_in_group)]
        
        idx = 0
        for i, client_id in enumerate(client_list):
            n_major_target = int(client_sizes[client_id] * 0.7)
            unique_alloc_size = min(label_client_alloc_sizes[i], n_major_target)
            chosen_major = label_samples.iloc[idx:idx + unique_alloc_size]
            
            # Nếu chưa đủ n_major_target, oversample with replacement từ label_samples
            current_len = len(chosen_major)
            if current_len < n_major_target:
                additional = n_major_target - current_len
                oversample = label_samples.sample(n=additional, replace=True, random_state=random_state)
                chosen_major = pd.concat([chosen_major, oversample])
            
            # Chỉ update used_idx với các samples unique
            unique_chosen = chosen_major.drop_duplicates(subset=chosen_major.columns.difference(['index'] if 'index' in chosen_major.columns else []))
            used_idx.update(unique_chosen.index)
            
            major_allocations[client_id] = chosen_major
            idx += unique_alloc_size  # Chỉ di chuyển idx theo unique alloc
    
    # Fill remain cho từng client
    clients = []
    for i in range(n_clients):
        chosen_major = major_allocations[i]
        n_remain = client_sizes[i] - len(chosen_major)
        remain_df = df[~df.index.isin(used_idx)]
        chosen_remain = remain_df.sample(n=min(n_remain, len(remain_df)), random_state=random_state) if n_remain > 0 and len(remain_df) > 0 else pd.DataFrame()
        used_idx.update(chosen_remain.index)
        client_df = pd.concat([chosen_major, chosen_remain]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        clients.append(client_df)
    
    return clients

# --- Đọc dữ liệu ---
# Thay đường dẫn này bằng file train.csv thực tế của bạn
csv_path = "../dataset/RSNA-ICH/binary_25k/train.csv"
df = pd.read_csv(csv_path)

# --- Chia dữ liệu ---
LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']
clients = split_label_majority(df, LABELS, n_clients=20)

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
plt.savefig("label_distribution_per_client.png")
plt.show()