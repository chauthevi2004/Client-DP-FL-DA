import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo dummy data cho RSNA-ICH (khoảng 20,000 samples, binary labels)
np.random.seed(42)
n_samples = 20000
labels = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']
df = pd.DataFrame({
    'ID': [f'ID_{i}' for i in range(n_samples)],
})
for label in labels[:-1]:  # Disease labels, rare ~5%
    df[label] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
df['healthy'] = 1 - df[labels[:-1]].max(axis=1)  # Healthy nếu không có disease

def split_label_majority_no_oversample_all_labels(df, labels, n_clients=20, random_state=42):
    """
    Chia đều sample cho các client, mỗi client ưu tiên 1 nhãn chính (label-majority).
    Đã chỉnh sửa để phân bổ samples của nhãn ưu tiên đều đặn trong nhóm client cùng ưu tiên một nhãn.
    Không oversampling.
    Đảm bảo mỗi client có đủ tất cả các loại nhãn bằng cách dự trữ và phân bổ ít nhất 1 sample cho mỗi nhãn thiếu.
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
    
    # Tìm major label cho từng client
    client_major = {}
    for label, client_list in groups.items():
        for cid in client_list:
            client_major[cid] = label
    
    # Dự trữ samples cho mỗi label để phân bổ cho các client không có major là label đó
    reserved = {label: pd.DataFrame() for label in labels}
    
    # Pre-allocate major samples cho từng client
    major_allocations = [pd.DataFrame() for _ in range(n_clients)]
    used_idx = set()
    
    for label, client_list in groups.items():
        if not client_list:
            continue
        label_samples = df[df[label] == 1].sample(frac=1, random_state=random_state)  # Shuffle samples
        available = len(label_samples)
        if available == 0:
            continue
        
        num_group = len(client_list)
        num_reserve = n_clients - num_group
        min_for_group = num_group  # Ít nhất 1 cho mỗi client trong group
        
        if available < num_reserve + min_for_group:
            reserve_size = max(0, available - min_for_group)
        else:
            reserve_size = num_reserve
        
        reserved[label] = label_samples.iloc[:reserve_size]
        major_samples = label_samples.iloc[reserve_size:]
        available_major = len(major_samples)
        
        # Phân bổ đều samples unique available cho các client trong nhóm
        base_per_client = available_major // num_group
        extra_per_client = available_major % num_group
        label_client_alloc_sizes = [base_per_client + (1 if i < extra_per_client else 0) for i in range(num_group)]
        
        idx = 0
        for i, client_id in enumerate(client_list):
            n_major_target = int(client_sizes[client_id] * 0.7)
            unique_alloc_size = min(label_client_alloc_sizes[i], n_major_target)
            chosen_major = major_samples.iloc[idx:idx + unique_alloc_size]
            
            major_allocations[client_id] = chosen_major
            used_idx.update(chosen_major.index)
            idx += unique_alloc_size  # Chỉ di chuyển idx theo unique alloc
    
    # Fill remain cho từng client, đảm bảo có ít nhất 1 sample cho mỗi label thiếu
    clients = []
    for i in range(n_clients):
        chosen_major = major_allocations[i]
        major_label = client_major.get(i, None)
        
        # Tính current counts
        current_counts = {l: chosen_major[l].sum() if l in chosen_major.columns else 0 for l in labels}
        
        missing_labels = [l for l in labels if current_counts[l] == 0]
        
        n_remain = client_sizes[i] - len(chosen_major)
        
        chosen_remain = pd.DataFrame()
        
        # Phân bổ ít nhất 1 cho mỗi missing label từ reserved hoặc remain
        for ml in missing_labels:
            one_sample = pd.DataFrame()
            if len(reserved[ml]) > 0:
                one_sample = reserved[ml].iloc[0:1]
                reserved[ml] = reserved[ml].drop(one_sample.index)
            else:
                # Nếu reserved hết, thử từ remain_df
                remain_df = df[~df.index.isin(used_idx)]
                candidates = remain_df[remain_df[ml] == 1]
                if len(candidates) > 0:
                    one_sample = candidates.sample(n=1, random_state=random_state)
            
            if not one_sample.empty:
                chosen_remain = pd.concat([chosen_remain, one_sample])
                used_idx.update(one_sample.index)
        
        # Fill phần còn lại ngẫu nhiên
        n_still_remain = n_remain - len(chosen_remain)
        remain_df = df[~df.index.isin(used_idx)]
        if n_still_remain > 0 and len(remain_df) > 0:
            additional = remain_df.sample(n=min(n_still_remain, len(remain_df)), random_state=random_state)
            chosen_remain = pd.concat([chosen_remain, additional])
            used_idx.update(additional.index)
        
        client_df = pd.concat([chosen_major, chosen_remain]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        clients.append(client_df)
    
    return clients

# --- Chia dữ liệu ---
clients = split_label_majority_no_oversample_all_labels(df, labels, n_clients=20)

# --- Tính phân phối nhãn và in ra ---
label_counts = []
for i, cdf in enumerate(clients):
    counts = [cdf[label].sum() for label in labels]
    label_counts.append(counts)
    print(f"Client {i}:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {int(count)}")
    print(f"  Total: {len(cdf)}\n")
label_counts = np.array(label_counts)

# --- Vẽ biểu đồ stacked bar để visualize tốt hơn ---
plt.figure(figsize=(14, 8))
bottom = np.zeros(len(clients))
for j, label in enumerate(labels):
    plt.bar(range(len(clients)), label_counts[:, j], bottom=bottom, label=label)
    bottom += label_counts[:, j]

plt.xlabel('Clients')
plt.ylabel('Number of Samples')
plt.title('Label Distribution per Client (Stacked Bar)')
plt.xticks(range(len(clients)))
plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("label_distribution_stacked.png")
plt.show()

# --- Heatmap cho phân phối ---
plt.figure(figsize=(12, 8))
sns.heatmap(label_counts.T, annot=True, fmt="d", cmap="YlGnBu", xticklabels=range(len(clients)), yticklabels=labels)
plt.title('Heatmap of Label Counts per Client')
plt.xlabel('Clients')
plt.ylabel('Labels')
plt.savefig("label_distribution_heatmap.png")
plt.show()

# --- Pie chart cho một client mẫu (Client 0) ---
plt.figure(figsize=(8, 8))
plt.pie(label_counts[0], labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Label Distribution in Client 0 (Pie Chart)')
plt.savefig("label_distribution_pie_client0.png")
plt.show()
