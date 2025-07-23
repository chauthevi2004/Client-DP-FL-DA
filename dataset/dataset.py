import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import random
import cv2
import time
import numpy as np
from PIL import Image
import os
import h5py
import scipy.io as scio
from glob import glob
import torch
from torch.utils.data import Dataset
import json
import pandas as pd

local = False if "uac/" in os.environ["HOME"] else True

LABELS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'healthy']

def get_domain(row):
    """
    Hàm này tính domain dựa trên NỘI DUNG Y KHOA của ảnh (loại bệnh).
    Nó sẽ không được sử dụng trong DFDataset nữa để đảm bảo mỗi client là một domain.
    """
    for i, label in enumerate(LABELS[:-1]):  # Exclude healthy first
        if row[label] == 1:
            return i
    return 5  # healthy if no disease

class BaseDataset(Dataset):
    def __init__(self, root_dir, images, labels, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(BaseDataset, self).__init__()
        self.root_dir = root_dir
        self.images = images
        self.labels = labels
        self.transform = transform
        self.site_idx = "base"

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]
        image_name = os.path.join(self.root_dir, self.images[index])
        # Giả định tên file ảnh cần thêm extension .png
        # Sửa đổi nếu tên file trong CSV đã có extension
        if not image_name.endswith('.png'):
            image_name += '.png'
            
        image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)

        return {
            "Raw_path": image_name,
            "Name": items,
            "Image": image,
            "Label": torch.tensor(label, dtype=torch.long), # Đảm bảo label là tensor
            "Site": self.site_idx,
        }

    def __len__(self):
        return len(self.images)

class DFDataset(BaseDataset):
    def __init__(self, root_dir, data_frame, transform=None, site_idx=None):
        """
        Args:
            data_dir: path to image directory.
            data_frame: data frame of image dirs and labels
            transform: optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.data_frame = data_frame
        # Giả định cột chứa ID ảnh là 'ImageID' và cột nhãn là 'healthy'
        self.images = data_frame["ImageID"].values
        self.labels = data_frame["healthy"].values.astype(np.int64)
        self.transform = transform
        super(DFDataset, self).__init__(
            root_dir=self.root_dir,
            images=self.images,
            labels=self.labels,
            transform=self.transform,
        )
        self.site_idx = site_idx

    # ===================================================================
    # SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY
    # ===================================================================
    def __getitem__(self, index):
        """
        Trả về item dữ liệu.
        Domain label được lấy trực tiếp từ `self.site_idx` để đảm bảo
        mỗi client được coi là một domain riêng biệt.
        """
        # Lấy item cơ bản (Image, Label, etc.) từ lớp cha
        item = super().__getitem__(index)
        
        # Kiểm tra xem site_idx có tồn tại không
        if self.site_idx is not None:
            # Gán domain label bằng ID của client (site_idx)
            item['Domain'] = torch.tensor(self.site_idx, dtype=torch.long)
            
        return item

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, client_idx, virtual_idx):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.client_idx = client_idx
        self.virtual_idx = virtual_idx

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class ProstateDataset(Dataset):
    def __init__(
        self,
        transform=None,
        site=0,
        split=0,
        splits=[0.5, 0.25, 0.25],
        seed=0,
        freerider=False,
        randrot=None,
    ):
        path = (
            "../data_preparation/dataset_2D"    # TODO: change the path here
            if not local
            else "../data_preparation/dataset_2D"
        )
        sites = {
            1: "I2CVB",
            2: "MSD",
            3: "NCI_ISBI_3T",
            4: "NCI_ISBI_DX",
            5: "Promise12",
            6: "ProstateX",
        }

        data_list = json.load(open(os.path.join('./dataset/Prostate', f"{sites[site]}.json")))
        names = ["training", "validation", "testing"]

        self.data_list = data_list[names[split]]
        if freerider:
            self.data_list = [self.data_list[0] for i in range(len(self.data_list))]
        self.randrot = randrot  # generating noisy clients by random rotation
        self.site_idx = site
        self.base_path = path
        self.client_idx = site - 1
        self.virtual_idx = 0
        names = ["Train", "Val", "Test"]
        print("=> loading site {}".format(site), names[split], flush=True)

        self.num = len(self.data_list)
        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        image_file, mask_file = sample["image"], sample["label"]

        with open(os.path.join(self.base_path, image_file), "rb") as f:
            image = Image.open(f).convert("RGB")
            image = np.asarray(image, dtype=np.float32)
            image = np.transpose(image, [2, 0, 1])
        with open(os.path.join(self.base_path, mask_file), "rb") as f:
            mask = Image.open(f).convert("L")
            if self.randrot is not None:
                mask = self.randrot(mask)
            mask = np.asarray(mask, dtype=np.float32)
            mask = np.expand_dims(mask, 0)

        data = {"Image": image / 255, "Mask": mask / 255}

        if self.transform is not None:
            data = self.transform(data)

            data["Image"].squeeze_(0)
            data["Mask"].squeeze_(0)

        return {
            "Raw_path": os.path.join(self.base_path, image_file),
            "Name": image_file,
            "Image": data["Image"],
            "Mask": data["Mask"],
            "Site": self.site_idx,
        }