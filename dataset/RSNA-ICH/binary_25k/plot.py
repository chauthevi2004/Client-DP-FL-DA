import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Hằng số và hàm chia dữ liệu từ code của bạn
LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']

def split_data_new_strategy(df, n_clients=20, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    
    specialist_groups = {i: [2*i, 2*i+1] for i in range(6)}
    clients = [pd.DataFrame() for _ in range(n_clients)]
    
    used_idx = set()
    for label_idx, label in enumerate(LABELS):
        if label == 'healthy':
            specialist_samples = df[df['healthy'] == 1]
        else:
            # Tìm các mẫu chỉ có duy nhất 1 nhãn bệnh
            specialist_samples = df[(df[label] == 1) & (df[LABELS].drop(label, axis=1).sum(axis=1) == 0) & (df['healthy'] == 0)]
        
        num_samples = len(specialist_samples)
        if num_samples == 0:
            continue
        
        # 60% dữ liệu chuyên biệt cho các client chuyên gia
        n_specialist = int(num_samples * 0.6)
        specialist_samples_shuffled = specialist_samples.sample(frac=1, random_state=random_state)
        allocated = specialist_samples_shuffled.iloc[:n_specialist]
        
        client1, client2 = specialist_groups[label_idx]
        half = n_specialist // 2
        clients[client1] = pd.concat([clients[client1], allocated.iloc[:half]])
        clients[client2] = pd.concat([clients[client2], allocated.iloc[half:n_specialist]])
        
        used_idx.update(allocated.index)
    
    # Dữ liệu còn lại (40% chuyên biệt + 100% đa nhãn)
    remaining_df = df[~df.index.isin(used_idx)]
    
    # Chia đều dữ liệu còn lại cho tất cả các client
    remaining_shuffled = remaining_df.sample(frac=1, random_state=random_state)
    chunk_size = len(remaining_shuffled) // n_clients
    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size if i < n_clients - 1 else len(remaining_shuffled)
        clients[i] = pd.concat([clients[i], remaining_shuffled.iloc[start:end]])
    
    # Shuffle lại lần cuối cho từng client
    for i in range(n_clients):
        clients[i] = clients[i].sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return clients

# --- Thực thi và Vẽ biểu đồ ---
# Tải dữ liệu training
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Vui lòng đảm bảo file 'train.csv' có trong cùng thư mục.")
    exit()

# Chia dữ liệu cho 20 clients
client_dfs = split_data_new_strategy(df_train, n_clients=20, random_state=42)

# Chuẩn bị dữ liệu để vẽ
counts = []
for i in range(20):
    # Đếm số lượng của từng loại nhãn cho mỗi client
    count_series = client_dfs[i][LABELS].sum()
    count_series.name = f'Client {i}'
    counts.append(count_series)

plot_df = pd.DataFrame(counts)

# Vẽ biểu đồ thanh xếp chồng
sns.set_style("whitegrid")
plot_df.plot(
    kind='bar', 
    stacked=True, 
    figsize=(18, 9),
    colormap='tab20' # Bảng màu tốt cho nhiều categories
)

plt.title('Phân phối dữ liệu trên 20 Clients', fontsize=20, fontweight='bold')
plt.xlabel('Client ID', fontsize=16)
plt.ylabel('Số lượng mẫu', fontsize=16)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Loại Nhãn', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
plt.tight_layout()

plt.savefig('output.png', dpi=300, bbox_inches='tight') 

plt.show()

print("Biểu đồ đã được lưu thành công vào file 'output.png'")
