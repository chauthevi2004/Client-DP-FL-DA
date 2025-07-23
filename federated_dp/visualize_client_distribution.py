import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from fed_train import LABELS, split_data_new_strategy


def plot_client_distribution(client_dfs, labels, title="Client Data Distribution", save_path=None):
    n_clients = len(client_dfs)
    label_counts = np.zeros((n_clients, len(labels)), dtype=int)
    for i, df in enumerate(client_dfs):
        for j, label in enumerate(labels):
            label_counts[i, j] = df[label].sum()
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    bottom = np.zeros(n_clients)
    colors = plt.cm.tab20.colors
    for j, label in enumerate(labels):
        ax.bar(np.arange(n_clients), label_counts[:, j], bottom=bottom, label=label, color=colors[j % len(colors)])
        bottom += label_counts[:, j]
    ax.set_xlabel('Client Index')
    ax.set_ylabel('Sample Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='../dataset/RSNA-ICH/binary_25k/train.csv', help='Path to CSV file')
    parser.add_argument('--n_clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str, default='output.png', help='Path to save the plot')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    client_dfs = split_data_new_strategy(df, n_clients=args.n_clients, random_state=args.seed)
    plot_client_distribution(client_dfs, LABELS, title=f"Client Data Distribution ({args.csv})", save_path=args.save)

if __name__ == "__main__":
    main() 