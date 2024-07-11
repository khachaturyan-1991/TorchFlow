import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import cv2
import glob
from utils import smart_resize
import re


class SmartDataset(Dataset):

    def __init__(self,
                 img_dir: str,
                 target_size: int = 128,
                 first_img: int = 0,
                 last_img: int = 100) -> None:
        super().__init__()
        self.img_paths = glob.glob(f"{img_dir}/*.png")[first_img:last_img]
        self.target_size = target_size

    def __len__(self):
        return len(self.img_paths)

    def _read_img(self, img_path, mask_path):
        img = cv2.imread(img_path)
        Y = cv2.imread(mask_path)
        img = smart_resize(img, target_size=self.target_size, channel=3)
        Y = smart_resize(Y, target_size=self.target_size, channel=3)
        img = torch.tensor(img)
        Y = torch.tensor(Y)
        img = img.to(torch.float32) / 255.0
        Y = Y.to(torch.float32)
        mask = torch.where(Y > 1, torch.ones_like(Y), Y)
        return img[:, :, 0:1].permute(2, 0, 1), mask[:, :, 0:1].permute(2, 0, 1)

    def __getitem__(self, index):
        pattern = r'(/images/)(img_)'
        replacement = r'/masks/seg_'
        img_path = self.img_paths[index]
        if isinstance(img_path, bytes):
            img_path = img_path.decode('utf-8')
        mask_path = re.sub(pattern, replacement, img_path)
        img, mask = self._read_img(img_path, mask_path)
        return img, mask


def create_dataloader(img_dir,
                      target_size=128,
                      batch_size=32,
                      shuffle=False,
                      first_img: int = 0,
                      last_img: int = 32):
    """
    Description
    -------
    Creates a dataloader from the custom dataset

    Parameters
    -------
    - image_dir: Path to the directory containing images.
    - masks_dir: Path to the directory containing masks.
    - target_size: Tuple (width, height) for resizing the images and masks.
    - batch_size: The size of the batches.
    - shuffle: Whether to shuffle the dataset.

    Returns
    -------
    - A tf.data.Dataset object.
    """
    dataset = SmartDataset(img_dir=img_dir,
                           target_size=target_size,
                           first_img=first_img, last_img=last_img)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    return dataloader
