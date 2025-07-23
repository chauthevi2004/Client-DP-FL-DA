import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import argparse
import time
import copy
import random
import math
import logging
import pandas as pd
import pickle as pkl
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision

from dataset.dataset import (
    ProstateDataset,
    DatasetSplit,
    DFDataset
)

from utils.util import setup_logger, get_timestamp
from utils.loss import DiceLoss, JointLoss
from nets.models import (
    DenseNet,
    UNet,
)

import torchvision.transforms as transforms
import monai.transforms as monai_transforms
from torchvision.transforms import ColorJitter, GaussianBlur, RandomRotation, Compose
from federated_dp.fedavg_trainer import FedAvgTrainer

from federated_dp.private_trainer import PrivateFederatedTrainer
from federated_dp.dp_adam_trainer import DPAdamTrainer
from federated_dp.dp2_rmsprop_trainer import DP2RMSPropTrainer
from federated_dp.fedadam_trainer import FedAdamTrainer
from federated_dp.fednova_trainer import FedNovaTrainer
from federated_dp.dp_nova_trainer import DPNovaTrainer
from federated_dp.fed2_rmsprop_trainer import Fed2RMSPropTrainer

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']

# Gradient Reversal Layer for Domain Adaptation (DANN)
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        
# Modified UNet with Domain Adaptation (DANN)
class UNetDA(UNet):
    def __init__(self, out_channels=1):
        super(UNetDA, self).__init__(out_channels=out_channels)
        # Domain classifier: assumes feature map from encoder
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, args.clients),
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, domain_label=None):
        features = self.encoder(x)  # Assuming encoder returns the bottleneck features
        seg_output = super().forward(x)  # Call original forward for segmentation

        if domain_label is not None:
            # Apply GRL and domain classification during training
            domain_features = self.grl(features)
            domain_output = self.domain_classifier(domain_features)
            return seg_output, domain_output
        return seg_output

# Modified DenseNet with Domain Adaptation (DANN)
class DenseNetDA(DenseNet):
    def __init__(self, num_classes=2, args=None):
        super(DenseNetDA, self).__init__(num_classes=num_classes)
        # Domain classifier: assumes features from the dense blocks
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, args.clients),  
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, domain_label=None):
        features = self.features(x)
        features = nn.ReLU()(features)
        features = nn.AdaptiveAvgPool2d((1, 1))(features)
        features = torch.flatten(features, 1)

        cls_output = self.classifier(features)

        if domain_label is not None:
            domain_features = self.grl(features)
            domain_output = self.domain_classifier(domain_features)
            return cls_output, domain_output
        return cls_output

def split_data_new_strategy(df, n_clients=20, random_state=42):
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Phân nhóm clients: 12 chuyên gia (2 per label 0-5)
    specialist_groups = {i: [2*i, 2*i+1] for i in range(6)}  # 0-11 specialists
    general_clients = list(range(12, 20))
    
    clients = [pd.DataFrame() for _ in range(n_clients)]
    
    # Bước 1: Xử lý ca chuyên biệt (chỉ 1 label=1, không multi, cho disease và healthy)
    used_idx = set()
    for label_idx, label in enumerate(LABELS):
        if label == 'healthy':
            specialist_samples = df[df['healthy'] == 1]
        else:
            specialist_samples = df[(df[label] == 1) & (df[LABELS].drop(label, axis=1).sum(axis=1) == 0) & (df['healthy'] == 0)]
        
        num_samples = len(specialist_samples)
        if num_samples == 0:
            continue
        
        # 60% cho chuyên gia
        n_specialist = int(num_samples * 0.6)
        specialist_samples_shuffled = specialist_samples.sample(frac=1, random_state=random_state)
        allocated = specialist_samples_shuffled.iloc[:n_specialist]
        
        # Chia đều cho 2 chuyên gia
        client1, client2 = specialist_groups[label_idx]
        half = n_specialist // 2
        clients[client1] = pd.concat([clients[client1], allocated.iloc[:half]])
        clients[client2] = pd.concat([clients[client2], allocated.iloc[half:n_specialist]])
        
        used_idx.update(allocated.index)
    
    # Dữ liệu còn lại: 40% chuyên biệt + multi-label
    remaining_df = df[~df.index.isin(used_idx)]
    
    # Shuffle và chia đều cho 20 clients
    remaining_shuffled = remaining_df.sample(frac=1, random_state=random_state)
    chunk_size = len(remaining_shuffled) // n_clients
    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size if i < n_clients - 1 else len(remaining_shuffled)
        clients[i] = pd.concat([clients[i], remaining_shuffled.iloc[start:end]])
    
    # Shuffle từng client df
    for i in range(n_clients):
        clients[i] = clients[i].sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return clients

def get_client_transforms(n_clients=20):
    transforms_list = []
    for i in range(n_clients):
        if i % 5 == 0:
            trans = ColorJitter(contrast=0.2 * ((i//5) % 3 + 1))
        elif i % 5 == 1:
            trans = GaussianBlur(kernel_size=3 + 2*(i//5 % 3), sigma=(0.1 + 0.2*(i//5 % 3), 1.0 + 0.5*(i//5 % 3)))
        elif i % 5 == 2:
            trans = ColorJitter(brightness=0.2 * ((i//5) % 3 + 1))
        elif i % 5 == 3:
            trans = RandomRotation(degrees=5 + 5*(i//5 % 3))
        else:
            trans = Compose([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), RandomRotation(10)])
        full_trans = Compose([
            transforms.Resize((224, 224)),
            trans,  # Biến đổi đặc trưng per client
            transforms.RandomAffine(degrees=20, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.12, contrast=0.15, saturation=0.12, hue=0.06),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transforms_list.append(full_trans)
    return transforms_list

def split_df(args, data_frame, num_users):
    print("Splitting data into {} users".format(num_users))
    df_list = np.array_split(data_frame.sample(frac=1, random_state=args.seed), num_users)
    return df_list


def split_dataset(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def balance_split_dataset(all_datasets, num_total_users):
    org_dataset_size = np.array([len(dataset) for dataset in all_datasets], dtype=np.int16)
    num_items = copy.deepcopy(org_dataset_size)
    num_splits = np.array([1 for _ in range(len(all_datasets))], dtype=np.int16)

    # split the client with maximum size, until the total number is achieved
    while num_splits.sum() < num_total_users:
        split_idx = np.argmax(num_items)
        num_splits[split_idx] += 1
        num_items[split_idx] = int(org_dataset_size[split_idx] / num_splits[split_idx])

    dict_users = [{} for _ in range(len(all_datasets))]
    for idx_site in range(len(all_datasets)):
        all_idxs = np.arange(org_dataset_size[idx_site])
        np.random.shuffle(all_idxs)

        for i in range(num_splits[idx_site]):
            if i == num_splits[idx_site] - 1:
                dict_users[idx_site][i] = set(all_idxs[i * num_items[idx_site] :])
            else:
                dict_users[idx_site][i] = set(
                    all_idxs[i * num_items[idx_site] : (i + 1) * num_items[idx_site]]
                )

    return dict_users


def initialize(args, logging):
    train_sites = []
    assert args.data in [
        "prostate",
        "RSNA-ICH",
    ]
    train_loaders, val_loaders, test_loaders = [], [], []
    trainsets, valsets, testsets = [], [], []
    if args.data == "prostate":
        assert args.clients <= 6
        if args.da:
            model = UNetDA(out_channels=1)
        else:
            model = UNet(out_channels=1)
        loss_fun = DiceLoss()
        domain_loss_fun = nn.CrossEntropyLoss()  # For domain adaptation loss
        # sites = ['BIDMC', 'HK',  'ISBI', 'ISBI_1.5', 'UCL']
        sites = [1, 2, 3, 4, 5, 6]
        train_sites = list(range(args.clients * args.virtual_clients))
        val_sites = [1, 2, 3, 4, 5, 6]
        keys = ["Image", "Mask"]
        data_splits = [0.6, 0.2, 0.2]
        train_data_sizes = []
        transform_list = [
            monai_transforms.Resized(keys, [256, 256]),
            monai_transforms.ToTensord(keys),
        ]

        transform = monai_transforms.Compose(transform_list)

        real_trainsets = []
        if args.generalize:
            if int(args.leave) in sites:
                leave_idx = sites.index(int(args.leave))
                sites.pop(leave_idx)
                generalize_sites = [int(args.leave)]
                logging.info("Source sites:" + str(sites))
                logging.info("Unseen sites:" + str(generalize_sites))
            else:
                raise ValueError(f"Unkown leave dataset{args.leave}")

            for site in sites:
                trainset = ProstateDataset(
                    transform=transform, site=site, split=0, splits=data_splits, seed=args.seed
                )
                valset = ProstateDataset(
                    transform=transform, site=site, split=1, splits=data_splits, seed=args.seed
                )
                logging.info(f"[Client {site}] Train={len(trainset)}, Val={len(valset)}")
                trainsets.append(trainset)
                valsets.append(valset)
            for site in generalize_sites:
                trainset = ProstateDataset(
                    transform=transform, site=site, split=0, splits=data_splits, seed=args.seed
                )
                valset = ProstateDataset(
                    transform=transform, site=site, split=1, splits=data_splits, seed=args.seed
                )
                testset = ProstateDataset(
                    transform=transform, site=site, split=2, splits=data_splits, seed=args.seed
                )
                wholeset = torch.utils.data.ConcatDataset([trainset, valset, testset])
                logging.info(f"[Unseen Client {site}] Test={len(wholeset)}")
                testsets.append(wholeset)
            real_trainsets = trainsets  # FIXED: Set real_trainsets for virtual splitting
            train_data_sizes = [len(s) for s in real_trainsets]  # FIXED: For possible client selection
            if args.clients < len(sites):
                idx = np.argsort(np.array(train_data_sizes))[::-1][: args.clients]
                real_trainsets = [real_trainsets[i] for i in idx]
                sites = [sites[i] for i in idx]
                valsets = [valsets[i] for i in idx]
                train_data_sizes = [train_data_sizes[i] for i in idx]
            if args.virtual_clients > 0:
                trainsets = []  # Reset trainsets for virtual
                if args.balance_split:
                    dict_users = balance_split_dataset(
                        real_trainsets, len(real_trainsets) * args.virtual_clients
                    )
                    for c_idx, client_trainset in enumerate(real_trainsets):
                        for v_idx in range(len(dict_users[c_idx])):
                            virtual_trainset = DatasetSplit(
                                client_trainset, dict_users[c_idx][v_idx], c_idx, v_idx
                            )
                            trainsets.append(virtual_trainset)
                            logging.info(
                                f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                            )
                else:
                    for c_idx, client_trainset in enumerate(real_trainsets):
                        dict_users = split_dataset(client_trainset, args.virtual_clients)
                        for v_idx in range(args.virtual_clients):
                            virtual_trainset = DatasetSplit(
                                client_trainset, dict_users[v_idx], c_idx, v_idx
                            )
                            trainsets.append(virtual_trainset)
                            logging.info(
                                f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                            )
            train_sites = list(range(len(trainsets)))  # FIXED: Update based on actual number
        else:
            for site in sites:
                if site == args.free:
                    trainset = ProstateDataset(
                        transform=transform,
                        site=site,
                        split=0,
                        splits=data_splits,
                        seed=args.seed,
                        freerider=True,
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed
                    )
                    logging.info(
                        f"[Free Rider Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                elif site == args.noisy:
                    trainset = ProstateDataset(
                        transform=transform,
                        site=site,
                        split=0,
                        splits=data_splits,
                        seed=args.seed,
                        randrot=transforms.RandomRotation(degrees=(1, 179)),
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed
                    )
                    logging.info(
                        f"[Noisy Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                else:
                    trainset = ProstateDataset(
                        transform=transform, site=site, split=0, splits=data_splits, seed=args.seed
                    )
                    valset = ProstateDataset(
                        transform=transform, site=site, split=1, splits=data_splits, seed=args.seed
                    )
                    testset = ProstateDataset(
                        transform=transform, site=site, split=2, splits=data_splits, seed=args.seed
                    )

                    logging.info(
                        f"[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
                    )
                train_data_sizes.append(len(trainset))
                real_trainsets.append(trainset)
                valsets.append(valset)
                testsets.append(testset)

            if args.merge:
                valset = torch.utils.data.ConcatDataset(valsets)
                testset = torch.utils.data.ConcatDataset(testsets)

            if not "no" in args.leave:
                if int(args.leave) in sites:
                    leave_idx = sites.index(int(args.leave))
                    sites.pop(leave_idx)
                    real_trainsets.pop(leave_idx)  # FIXED: Pop real_trainsets instead of trainsets
                    valsets.pop(leave_idx)
                    testsets.pop(leave_idx)
                    train_data_sizes.pop(leave_idx)  # FIXED: Pop train_data_sizes
                    logging.info("New sites:" + str(sites))
                    generalize_sites = [int(args.leave)]
                else:
                    raise ValueError(f"Unkown leave dataset{args.leave}")

            if args.clients < 6:
                idx = np.argsort(np.array(train_data_sizes))[::-1][: args.clients]
                real_trainsets = [real_trainsets[i] for i in idx]
                sites = [sites[i] for i in idx]
                valsets = [valsets[i] for i in idx]
                testsets = [testsets[i] for i in idx]  # FIXED: Pop testsets too if merge not set yet

            if args.virtual_clients > 0:
                if args.balance_split:
                    dict_users = balance_split_dataset(
                        real_trainsets, len(sites) * args.virtual_clients
                    )
                    for c_idx, client_trainset in enumerate(real_trainsets):
                        for v_idx in range(len(dict_users[c_idx])):
                            virtual_trainset = DatasetSplit(
                                client_trainset, dict_users[c_idx][v_idx], c_idx, v_idx
                            )
                            trainsets.append(virtual_trainset)
                            logging.info(
                                f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                            )

                else:
                    for c_idx, client_trainset in enumerate(real_trainsets):
                        dict_users = split_dataset(client_trainset, args.virtual_clients)
                        for v_idx in range(args.virtual_clients):
                            virtual_trainset = DatasetSplit(
                                client_trainset, dict_users[v_idx], c_idx, v_idx
                            )
                            trainsets.append(virtual_trainset)
                            logging.info(
                                f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                            )
            else:
                trainsets = real_trainsets  # FIXED: Set trainsets if no virtual
            train_sites = list(range(len(trainsets)))  # FIXED: Update
    elif args.data == "RSNA-ICH":
        N_total_client = 20
        assert args.clients <= N_total_client
        sites = list(range(N_total_client))
        generalize_sites = None

        if args.da:
            model = DenseNetDA(num_classes=2, args=args)
        else:
            model = DenseNet(num_classes=2)
        if args.pretrain:
            model_dict = model.state_dict()
            pretrained_dict = model_zoo.load_url(
                "https://download.pytorch.org/models/densenet121-a639ec97.pth"
            )
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if (k in model_dict and "classifier" not in k and "domain_classifier" not in k)
            }
            model.load_state_dict(pretrained_dict, strict=False)
        loss_fun = nn.CrossEntropyLoss()  # Dùng CrossEntropyLoss mặc định, sẽ truyền class_weights khi train
        domain_loss_fun = nn.CrossEntropyLoss()  # For domain adaptation loss
        train_sites = list(range(args.clients * args.virtual_clients))  # virtual clients
        val_sites = list(range(N_total_client))  # original clients
        train_data_sizes = []
        ich_folder = "binary_25k"

        train_df = pd.read_csv(f"../dataset/RSNA-ICH/{ich_folder}/train.csv")
        train_dfs = split_data_new_strategy(train_df, n_clients=N_total_client, random_state=args.seed)
        
        val_df = pd.read_csv(f"../dataset/RSNA-ICH/{ich_folder}/validate.csv")
        val_dfs = split_data_new_strategy(val_df, n_clients=N_total_client, random_state=args.seed)
        
        test_df = pd.read_csv(f"../dataset/RSNA-ICH/{ich_folder}/test.csv")
        test_dfs = split_data_new_strategy(test_df, n_clients=N_total_client, random_state=args.seed)

        """split trainsets for virtual clients"""
        client_transforms = get_client_transforms(N_total_client)
        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        real_trainsets = []
        valsets_all = []
        testsets_all = []
        class_weights_all = []
        for idx in range(N_total_client):
            trainset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",    
                data_frame=train_dfs[idx],
                transform=client_transforms[idx],
                site_idx=idx,
            )
            valset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",
                data_frame=val_dfs[idx],
                transform=transform_test,
                site_idx=idx,
            )
            testset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",
                data_frame=test_dfs[idx],
                transform=transform_test,
                site_idx=idx,
            )
            logging.info(
                f"[Client {idx}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}"
            )
            train_data_sizes.append(len(trainset))
            real_trainsets.append(trainset)
            valsets_all.append(valset)
            testsets_all.append(testset)
            # Compute class weights for this client
            df = train_dfs[idx]
            num_healthy = (df['healthy'] == 1).sum()
            num_disease = len(df) - num_healthy
            if num_healthy == 0:
                num_healthy = 1
            if num_disease == 0:
                num_disease = 1
            total = len(df)
            weight_healthy = total / (2.0 * num_healthy)
            weight_disease = total / (2.0 * num_disease)
            class_weights_all.append(torch.tensor([weight_healthy, weight_disease], dtype=torch.float32))

        # Sửa lỗi: Đảm bảo trainsets luôn được gán từ real_trainsets
        trainsets = real_trainsets
        valsets = valsets_all
        testsets = testsets_all
        class_weights = class_weights_all

        # Sửa lỗi: Nếu số client mong muốn < N_total_client, chỉ giữ lại đúng số lượng client
        if args.clients < N_total_client:
            import numpy as np
            selected_client_indices = np.random.choice(N_total_client, args.clients, replace=False)
            selected_client_indices = sorted(selected_client_indices)
            trainsets = [real_trainsets[i] for i in selected_client_indices]
            valsets = [valsets_all[i] for i in selected_client_indices]
            testsets = [testsets_all[i] for i in selected_client_indices]
            class_weights = [class_weights_all[i] for i in selected_client_indices]
            sites = selected_client_indices  # FIXED: Update sites
            train_sites = list(range(args.clients))
            val_sites = [val_sites[i] for i in selected_client_indices]
        else:
            trainsets = real_trainsets
            valsets = valsets_all
            testsets = testsets_all
            class_weights = class_weights_all
            train_sites = list(range(N_total_client))

        if args.merge:
            valset = torch.utils.data.ConcatDataset(valsets)
            testset = torch.utils.data.ConcatDataset(testsets)

        if args.generalize and args.leave is not None and args.leave != "no":
            # LOCO: Loại client leave_idx khỏi train/val, chỉ test trên client đó
            leave_idx = int(args.leave)
            assert 0 <= leave_idx < N_total_client
            selected_client_indices = [i for i in range(N_total_client) if i != leave_idx]
            sites = selected_client_indices
            generalize_sites = [leave_idx]
            train_sites = list(range(len(selected_client_indices)))
            # Lọc lại các tập dữ liệu và class_weights cho train/val
            real_trainsets = [real_trainsets[i] for i in selected_client_indices]
            trainsets = real_trainsets  # FIXED: Update trainsets after filtering
            valsets = [valsets_all[i] for i in selected_client_indices]
            testsets = [testsets_all[i] for i in selected_client_indices]
            class_weights = [class_weights_all[i] for i in selected_client_indices]
            val_sites = [val_sites[i] for i in selected_client_indices]
            client_domain_map = {i: i for i in range(len(selected_client_indices))}
            # Log mapping client-domain
            for i, idx in enumerate(selected_client_indices):
                print(f"[RSNA-ICH] Train/Val Client {i} (gốc idx {idx}) là domain riêng biệt (domain index {i})")
                logging.info(f"[RSNA-ICH] Train/Val Client {i} (gốc idx {idx}) là domain riêng biệt (domain index {i})")
            # FIXED: Chuẩn bị testset cho domain unseen using wholeset (train + val + test of unseen client), matching Prostate logic
            unseen_trainset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",
                data_frame=train_dfs[leave_idx],
                transform=transform_test,
                site_idx=leave_idx,
            )
            unseen_valset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",
                data_frame=val_dfs[leave_idx],
                transform=transform_test,
                site_idx=leave_idx,
            )
            unseen_testset = DFDataset(
                root_dir="../data/RSNA-ICH/organized/stage_2_train",
                data_frame=test_dfs[leave_idx],
                transform=transform_test,
                site_idx=leave_idx,
            )
            wholeset = torch.utils.data.ConcatDataset([unseen_trainset, unseen_valset, unseen_testset])
            logging.info(f"[Unseen Client {leave_idx}] Test={len(wholeset)}")
            unseen_testsets = [wholeset]
            testsets = unseen_testsets  # FIXED: Use wholeset for unseen testsets
            print(f"[RSNA-ICH] Unseen test client: {leave_idx}")
            logging.info(f"[RSNA-ICH] Unseen test client: {leave_idx}")
        else:
            if args.clients < N_total_client:
                np.random.seed(args.seed)
                selected_client_indices = np.random.choice(N_total_client, args.clients, replace=False)
                selected_client_indices = sorted(selected_client_indices)
                real_trainsets = [real_trainsets[i] for i in selected_client_indices]
                trainsets = real_trainsets  # FIXED
                valsets = [valsets_all[i] for i in selected_client_indices]
                testsets = [testsets_all[i] for i in selected_client_indices]
                class_weights = [class_weights_all[i] for i in selected_client_indices]
                val_sites = [val_sites[i] for i in selected_client_indices]
                client_domain_map = {i: i for i in range(len(selected_client_indices))}
                train_sites = list(range(len(selected_client_indices)))
                for i, idx in enumerate(selected_client_indices):
                    print(f"[RSNA-ICH] Client {i} (gốc idx {idx}) là domain riêng biệt (domain index {i})")
                    logging.info(f"[RSNA-ICH] Client {i} (gốc idx {idx}) là domain riêng biệt (domain index {i})")
            else:
                client_domain_map = {i: i for i in range(N_total_client)}
                full_class_weights = class_weights
                train_sites = list(range(N_total_client))

        # FIXED: Add virtual clients splitting for RSNA in both generalize and non-generalize
        real_trainsets = trainsets
        if args.virtual_clients > 0:
            trainsets = []
            if args.balance_split:
                dict_users = balance_split_dataset(
                    real_trainsets, len(real_trainsets) * args.virtual_clients
                )
                for c_idx, client_trainset in enumerate(real_trainsets):
                    for v_idx in range(len(dict_users[c_idx])):
                        virtual_trainset = DatasetSplit(
                            client_trainset, dict_users[c_idx][v_idx], c_idx, v_idx
                        )
                        trainsets.append(virtual_trainset)
                        logging.info(
                            f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                        )
            else:
                for c_idx, client_trainset in enumerate(real_trainsets):
                    dict_users = split_dataset(client_trainset, args.virtual_clients)
                    for v_idx in range(args.virtual_clients):
                        virtual_trainset = DatasetSplit(
                            client_trainset, dict_users[v_idx], c_idx, v_idx
                        )
                        trainsets.append(virtual_trainset)
                        logging.info(
                            f"[Virtual Client {c_idx}-{v_idx}] Train={len(virtual_trainset)}"
                        )
        train_sites = list(range(len(trainsets)))  # FIXED: Update train_sites based on actual

    else:
        train_sites = []
        raise NotImplementedError
    if args.debug:
        trainsets = [
            torch.utils.data.Subset(trset, list(range(args.batch * 4))) for trset in trainsets
        ]
        if args.merge:
            valset = torch.utils.data.Subset(valset, list(range(args.batch * 2)))
            testset = torch.utils.data.Subset(testset, list(range(args.batch * 2)))
        else:
            valsets = [
                torch.utils.data.Subset(trset, list(range(args.batch * 4)))
                for trset in valsets[: len(valsets)]
            ]
            testsets = [
                torch.utils.data.Subset(trset, list(range(args.batch * 4)))
                for trset in testsets[: len(testsets)]
            ]

    if args.balance:
        assert args.split == "FeatureNonIID"
        min_data_len = min([len(s) for s in trainsets])
        print(f"Balance training set, using {args.percent*100}% training data")
        for idx in range(len(trainsets)):
            trainset = torch.utils.data.Subset(
                trainsets[idx], list(range(int(min_data_len * args.percent)))
            )
            print(f"[Client {sites[idx]}] Train={len(trainset)}")

            train_loaders.append(
                torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True)
            )
        if args.merge:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch, shuffle=False
            )
        else:
            for idx in range(len(valsets)):
                valset = valsets[idx]
                testset = testsets[idx]
                val_loaders.append(
                    torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
                )
                test_loaders.append(
                    torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)
                )
    else:
        for idx in range(len(trainsets)):
            if args.debug:
                train_loaders.append(
                    torch.utils.data.DataLoader(
                        trainsets[idx], batch_size=args.batch, shuffle=False, drop_last=False
                    )
                )
            else:
                train_loaders.append(
                    torch.utils.data.DataLoader(
                        trainsets[idx], batch_size=args.batch, shuffle=True, drop_last=True
                    )
                )
        if args.merge:
            val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=args.batch, shuffle=False
            )
        else:
            for idx in range(len(valsets)):
                valset = valsets[idx]
                val_loaders.append(
                    torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False)
                )
            for idx in range(len(testsets)):
                testset = testsets[idx]
                test_loaders.append(
                    torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False)
                )

    if args.generalize:
        if args.merge:
            return (
                model,
                loss_fun,
                domain_loss_fun if args.da else None,
                sites,
                generalize_sites,
                trainsets,
                valsets,
                testsets,
                train_loaders,
                val_loaders,
                test_loaders,
            )
        else:
            return (
                model,
                loss_fun,
                domain_loss_fun if args.da else None,
                sites,
                generalize_sites,
                trainsets,
                valsets,
                testsets,
                train_loaders,
                val_loaders,
                test_loaders,
            )
    else:
        if args.merge:
            return (
                model,
                loss_fun,
                domain_loss_fun if args.da else None,
                train_sites,
                val_sites,
                trainsets,
                valsets,
                testsets,
                train_loaders,
                val_loader,
                test_loader,
            )
        else:
            return (
                model,
                loss_fun,
                domain_loss_fun if args.da else None,
                train_sites,
                val_sites,
                trainsets,
                valsets,
                testsets,
                train_loaders,
                val_loaders,
                test_loaders,
            )


if __name__ == "__main__":
    os.environ["TORCH_HOME"] = "../../torchhome"

    parser = argparse.ArgumentParser()
    # Federated training settings
    parser.add_argument("-N", "--clients", help="The number of participants", type=int, default=10)
    parser.add_argument(
        "-VN", "--virtual_clients", help="The number of virtual clients", type=int, default=1
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=-1, help="learning rate decay for scheduler"
    )
    parser.add_argument(
        "--early", action="store_true", help="early stop w/o improvement over 20 epochs"
    )
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--rounds", type=int, default=200, help="iterations for communication")
    parser.add_argument("--local_epochs", type=int, default=1, help="local training epochs")
    parser.add_argument("--mode", type=str, default="fedavg", help="different FL algorithms")
    parser.add_argument(
        "--pretrain", action="store_true", help="Use Alexnet/ResNet pretrained on Imagenet"
    )
    # Experiment settings
    parser.add_argument("--exp", type=str, default=None, help="exp name")
    parser.add_argument(
        "--save_path", type=str, default="../checkpoint/", help="path to save the checkpoint"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from the save path checkpoint"
    )
    parser.add_argument("--gpu", type=str, default="0", help='gpu device number e.g., "0,1,2"')
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    # Data settings
    parser.add_argument(
        "--data", type=str, default="digits5", help="Different dataset: cifar10, cifar10c"
    )
    parser.add_argument(
        "-sr", "--sample_rate", type=float, default=1, help="Sample rate at each round"
    )
    parser.add_argument("--leave", type=str, default="no", help="leave one domain/client out")
    parser.add_argument("--merge", action="store_true", help="Use a global val/test set")
    parser.add_argument("--balance", action="store_true", help="Do not balance training data")
    parser.add_argument(
        "--weighted_avg",
        action="store_true",
        help="Use weighted average, default is pure avg, i.e., 1/N",
    )
    # CIFAR-10 Split
    parser.add_argument(
        "--split",
        type=str,
        default="UNI",
        choices=["UNI", "POW", "LabelNonIID", "FeatureNonIID"],
        help="Data distribution setting",
    )
    # Method settings
    parser.add_argument(
        "--local_bn", action="store_true", help="Do not aggregate BN during communication"
    )
    parser.add_argument("--generalize", action="store_true", help="Generalization setting")
    parser.add_argument("--gn", action="store_true", help="use groupnorm")
    parser.add_argument("--selu", action="store_true", help="use_selu")
    parser.add_argument(
        "--comb",
        type=str,
        default="times",
        choices=["times", "plus"],
        help="Combination mode, cos+lval or cosxlval",
    )
    parser.add_argument(
        "--free",
        type=int,
        default=-1,
        help="Set a client as free rider (always providing repeating data)",
    )
    parser.add_argument("--noisy", type=int, default=-1, help="Set a client as a noisy client")
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="The hyper parameter for tune loss for DC"
    )
    parser.add_argument(
        "--adaclip",
        action="store_true",
        help="use adaptive clip (meadian norm as norm clip bound)",
    )
    parser.add_argument(
        "--noclip",
        action="store_true",
        help="Do not clip gradients, only for ideal trial",
    )

    parser.add_argument("--ema", type=float, default=0.0, help="the rate for keeping history")

    # FL algorithm hyper parameters
    parser.add_argument("--mu", type=float, default=1e-3, help="The hyper parameter for fedprox")
    parser.add_argument("--S", type=float, default=10, help="The hyper parameter for dp")
    # parser.add_argument("--sigma", type=float, default=1e-4, help="The hyper parameter for dp")

    parser.add_argument("--epsilon", type=float, default=None, help="The budget for dp")
    parser.add_argument("--noise_multiplier", type=float, default=None, help="The budget for dp")
    parser.add_argument("--delta", type=float, default=1e-3, help="The budget for dp")
    parser.add_argument("--accountant", type=str, default="prv", help="The dp accountant")
    parser.add_argument(
        "--dp_mode",
        type=str,
        default="overhead",
        choices=["overhead", "bounded"],
        help="Using which mode to do private training. Options: overhead, bounded.",
    )
    parser.add_argument(
        "--balance_split", action="store_true", help="Activatebalance virtual client splitting."
    )
    parser.add_argument("--test", action="store_true", help="Running test mode.")
    parser.add_argument("--ckpt", type=str, default="None", help="Path for the testing ckpt")
    parser.add_argument(
        "--adam_lr", type=float, default=0.1, help="Global learning rate for FedAdam."
    )
    parser.add_argument("--dp2_interval", type=int, default=3, help="Interval for DP2-RMSProp")
    parser.add_argument(
        "--rmsprop_lr", type=float, default=1, help="Global learning rate for DP2-RMSProp."
    )
    parser.add_argument(
        "--ada_vn",
        action="store_true",
        help="Running adaptive virtual client splitting. VN = ceil(vn*self.virtual_clients), vn may vary in range [-2,2]",
    )
    parser.add_argument(
        "--init_vn",
        action="store_true",
        help="Esitimating virtual client splitting using first round results.",
    )
    parser.add_argument(
        "--ada_stable",
        action="store_true",
        help="VN = ceil(vn)*self.virtual_clients, vn should be unchanged",
    )
    parser.add_argument(
        "--ada_prog",
        action="store_true",
        help="VN = ceil(vn/2 * self.virtual_clients), progressively reaching optimal VN",
    )
    parser.add_argument("--da", action="store_true", help="Enable Domain Adaptation (DANN)")

    args = parser.parse_args()
    trainer_dict = {
        "fedavg": FedAvgTrainer,
        "fedadam": FedAdamTrainer,
        "fednova": FedNovaTrainer,
        "fed2rmsprop": Fed2RMSPropTrainer,
        "dpsgd": PrivateFederatedTrainer,
        "dpadam": DPAdamTrainer,
        "dpnova": DPNovaTrainer,
        "dp2rmsprop": DP2RMSPropTrainer,
    }
    assert args.mode in trainer_dict.keys()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    code_folder = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    code_folder = "v1.0"

    if args.generalize:
        args.save_path = "../../experiments/{}/checkpoint/{}_generalize/seed{}".format(
            code_folder, args.data, seed
        )
    else:
        args.save_path = "../../experiments/{}/checkpoint/{}/seed{}".format(
            code_folder, args.data, seed
        )

    if "dp" in args.mode or "our" in args.mode:
        if args.dp_mode == "overhead":
            assert args.epsilon is not None or args.noise_multiplier is not None

            if args.epsilon is not None:
                dp_info = f"overhead_epsilon{args.epsilon}_delta{args.delta}"
            else:
                dp_info = f"overhead_z{args.noise_multiplier}_delta{args.delta}"

        elif args.dp_mode == "bounded":
            assert args.epsilon is not None and args.noise_multiplier is not None
            dp_info = f"bounded_epsilon{args.epsilon}_z{args.noise_multiplier}_delta{args.delta}"
        else:
            raise NotImplementedError

        if args.adaclip:
            dp_info += "_adaclip"
        else:
            if args.noclip:
                dp_info += "_Noclip"
            else:
                dp_info += f"_S{args.S}"
    else:
        dp_info = ""

    exp_folder = (
        "{}_rounds{}_localE{}_lr{}_batch{}_N{}_sr{}_VN{}_{}".format(
            args.mode,
            args.rounds,
            args.local_epochs,
            args.lr,
            args.batch,
            args.clients,
            args.sample_rate,
            args.virtual_clients,
            dp_info,
        )
        if args.exp is None
        else args.exp
    )

    if args.mode == "fedadam" or args.mode == "dpadam":
        exp_folder = exp_folder + f"_adamlr={args.adam_lr}"
    if args.mode == "fed2rmsprop" or args.mode == "dp2rmsprop":
        exp_folder = exp_folder + f"_rmsproplr={args.rmsprop_lr}"

    if args.merge:
        exp_folder = exp_folder + "_MergeValTest"
    if args.weighted_avg:
        exp_folder = exp_folder + "_WeightedAvg"
    else:
        exp_folder = exp_folder + "_PureAvg"

    if args.gn:
        exp_folder = exp_folder + "_GN"
    if args.selu:
        exp_folder = exp_folder + "_SELU"
    
    if args.pretrain:
        exp_folder = exp_folder + "_pretrained"
    if args.early:
        exp_folder = exp_folder + "_early_stop"
    if args.lr_decay > 0:
        exp_folder = exp_folder + f"_lrdecay{args.lr_decay}"

    if args.balance:
        exp_folder = exp_folder + "_balance_train"
    if args.generalize or not "no" in args.leave:
        exp_folder = exp_folder + "_leave_" + args.leave

    if args.balance_split:
        exp_folder = exp_folder + "_balance_split"

    if args.ema > 0.0:
        exp_folder = exp_folder + f"_ema{args.ema}"

    if args.ada_vn:
        exp_folder = exp_folder + "_adaVN"
    if args.ada_stable:
        exp_folder = exp_folder + "_adaStable"
    if args.ada_prog:
        exp_folder = exp_folder + "_adaProg"

    if args.init_vn:
        exp_folder = exp_folder + "_InitVN"
    if args.debug:
        exp_folder = exp_folder + "_debug"
    if args.test:
        exp_folder = exp_folder + "_test"
    if args.da:
        exp_folder = exp_folder + "_DA"

    args.save_path = os.path.join(args.save_path, exp_folder)
    if not args.test:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    SAVE_PATH = args.save_path

    # setup the logger
    log_path = args.save_path.replace("/checkpoint/", "/log/")
    args.log_path = log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    lg = setup_logger(
        f"{args.mode}-{get_timestamp()}",
        log_path,
        level=logging.INFO,
        screen=False,
        tofile=True,
    )

    lg = logging.getLogger(f"{args.mode}-{get_timestamp()}")


    if args.generalize:
        lg.info("Generalize setting")
    if args.generalize:
        (
            server_model,
            loss_fun,
            domain_loss_fun,
            train_sites,  # FIXED: Changed from datasets to train_sites for consistency
            generalize_sites,
            train_sets,
            val_sets,
            test_sets,
            train_loaders,
            val_loaders,
            test_loaders,
        ) = initialize(args, lg)
        val_sites = train_sites  # FIXED: Set val_sites to train_sites in generalize
    else:
        generalize_sites = None
        (
            server_model,
            loss_fun,
            domain_loss_fun,
            train_sites,
            val_sites,
            train_sets,
            val_sets,
            test_sets,
            train_loaders,
            val_loaders,
            test_loaders,
        ) = initialize(args, lg)

    # Sau khi chia train_sets và loại leave (nếu có), cập nhật lại số client thực tế
    args.clients = len(train_sets)
    assert (
        int(args.clients * args.virtual_clients) == len(train_loaders) == len(train_sites)
    ), f"Virtual client num {args.clients * args.virtual_clients}, train loader num {len(train_loaders)}, train site num {len(train_sites)} do not match."
    assert len(val_loaders) == len(val_sites)  # == int(args.clients)
    train_total_len = sum([len(tr_set) for tr_set in train_sets])
    client_weights = (
        [len(tr_set) / train_total_len for tr_set in train_sets]
        if args.weighted_avg
        else [
            1.0 / float(int(args.clients * args.sample_rate) * args.virtual_clients)
            for i in range(int(args.clients * args.virtual_clients))
        ]
    )
    lg.info("Client Weights: " + str(client_weights))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("#Device:", device)
    if args.test:
        logging.info("Using checkpoint: {}".format(args.ckpt))
        logging.info("Testing Clients:{}".format(val_sites))
    else:
        lg.info("Training Clients:{}".format(train_sites))
        lg.info("Val Clients:{}".format(val_sites))
    lg.info("=============== args ================")
    lg.info(str(args))

    # setup the summarywriter
    from torch.utils.tensorboard import SummaryWriter

    args.writer = SummaryWriter(log_path)

    # Đảm bảo full_class_weights luôn tồn tại
    if 'full_class_weights' not in locals():
        full_class_weights = None

    # setup trainer
    TrainerClass = trainer_dict[args.mode]

    trainer = TrainerClass(
        args,
        lg,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=client_weights,
        generalize_sites=generalize_sites,
        class_weights=full_class_weights,
        client_domain_map=client_domain_map if 'client_domain_map' in locals() else None,
    )

    trainer.best_changed = False
    trainer.early_stop = 20

    trainer.client_steps = [torch.tensor(len(train_loader)) for train_loader in train_loaders]
    print("Client steps:", trainer.client_steps)

    if args.resume:
        checkpoint = torch.load(os.path.join(SAVE_PATH, args.ckpt))
        trainer.server_model.load_state_dict(checkpoint["server_model"])
        if args.local_bn:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(
                    checkpoint["model_{}".format(client_idx)]
                )
        else:
            for client_idx in range(trainer.client_num):
                trainer.client_models[client_idx].load_state_dict(checkpoint["server_model"])
        trainer.best_epoch, trainer.best_acc = checkpoint["best_epoch"], checkpoint["best_acc"]
        trainer.start_iter = int(checkpoint["a_iter"]) + 1

        print("Resume training from epoch {}".format(trainer.start_iter))
    else:
        # log the best for each model on all datasets
        trainer.best_epoch = 0
        trainer.best_acc = 0.0
        trainer.start_iter = 0

    if args.test:
        if args.generalize and args.leave is not None and args.leave != "no":
            trainer.inference(args.ckpt, test_loaders, loss_fun, generalize_sites, process=True)  # FIXED: Use test_loaders and generalize_sites
        else:
            trainer.inference(args.ckpt, test_loaders, loss_fun, val_sites, process=True)
    else:
        trainer.start(
            train_loaders, val_loaders, test_loaders, loss_fun, SAVE_PATH, generalize_sites, domain_loss_fun=domain_loss_fun if args.da else None
        )

    logging.shutdown()