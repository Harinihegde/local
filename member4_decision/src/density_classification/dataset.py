import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image

from config import DENSITY_THRESHOLDS, IMAGENET_MEAN, IMAGENET_STD


def count_people_from_gt(gt_path):
    try:
        mat = sio.loadmat(gt_path)
        points = mat.get('image_info', [[]])[0, 0][0, 0][0]
        return len(points)
    except:
        return 0


def assign_density_label(count):
    if count < DENSITY_THRESHOLDS['sparse'][1]:
        return 0  # Sparse
    elif count < DENSITY_THRESHOLDS['medium'][1]:
        return 1  # Medium
    else:
        return 2  # Dense


def process_folder(img_folder, gt_folder, split):
    records = []
    img_files = sorted(Path(img_folder).glob('*.jpg'))
    for img_path in tqdm(img_files, desc=f"Processing {split}"):
        gt_filename = img_path.stem.replace('IMG', 'GT_IMG') + '.mat'
        gt_path = Path(gt_folder) / gt_filename
        if gt_path.exists():
            person_count = count_people_from_gt(str(gt_path))
            label = assign_density_label(person_count)
            records.append({
                'image_path': str(img_path),
                'person_count': person_count,
                'density_label': label,
                'density_class': ['Sparse', 'Medium', 'Dense'][label],
                'split': split
            })
    return records


def build_dataframe(data_root):
    from pathlib import Path
    data_records = []
    splits = [
        (f'{data_root}/part_A/train_data', 'train'),
        (f'{data_root}/part_A/test_data', 'test'),
        (f'{data_root}/part_B/train_data', 'train'),
        (f'{data_root}/part_B/test_data', 'test'),
    ]
    for base, split in splits:
        img_folder = Path(base) / 'images'
        gt_folder  = Path(base) / 'ground-truth'
        if img_folder.exists():
            data_records += process_folder(img_folder, gt_folder, split)
    return pd.DataFrame(data_records)


class DensityDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row['density_label'], dtype=torch.long)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transform, val_transform