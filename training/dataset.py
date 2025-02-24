import os
import lmdb
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.helper import parse_int_list
from pathlib import Path
from PIL import Image
import ehtim as eh


class ImageFolder(Dataset):
    def __init__(self, root, 
                 id_list=None,           # string, e.g., '0-9,2-5'
                 resolution=256,
                 num_channels=3, 
                 img_ext='png'):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.num_channels = num_channels
        self.resizer = transforms.Resize((resolution, resolution))
        id_list = parse_int_list(id_list)
        if id_list is None:
            # search for all images in the folder
            # Define the file extensions to search for
            extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
            img_path_list = [file for ext in extensions for file in Path(root).rglob(ext)]
            img_path_list = sorted(img_path_list)
            self.id2path = {i: img_path for i, img_path in enumerate(img_path_list)}
            self.length = len(img_path_list)
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.id2path = {i: os.path.join(self.root, f'{str(id).zfill(5)}.{img_ext}') for i, id in enumerate(id_list)}
            self.length = len(id_list)
            self.id_list = id_list
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.id2path[idx]
        img = self.load_raw_image(img_path)
        img = self.normalize(img)
        img = torch.from_numpy(img).to(torch.float32)
        if img.shape[-1] != self.resolution:
            img = self.resizer(img)
        return img
    
    def save_image(self, img, img_path):
        '''
        Save the image.
        Args:
            - img: image, (C, H, W), ndarray, np.uint8.
            - img_path: path to save the image, str.
        '''
        img = img.transpose(1, 2, 0)    # (C, H, W) -> (H, W, C)
        img = Image.fromarray(img)
        img.save(img_path)


    def load_raw_image(self, img_path):
        '''
        Load the image and convert it to CHW format.
        Args:
            - img_path: path to the image, str.
        Returns:
            - img: image, (C, H, W), ndarray, np.uint8.
        '''
        img = np.array(Image.open(img_path))
        img = img.transpose(2, 0, 1)    # (H, W, C) -> (C, H, W)
        return img


    def normalize(self, img):
        '''
        Normalize the image to [-1, 1].
        Args:
            - img: image, (C, H, W), numpy array.
        Returns:
            - img: image, (C, H, W), numpy array.
        '''
        img = img / 127.5 - 1.0
        return img


    def unnormalize(self, img):
        '''
        Normalize the image to [0, 1]
        Args:
            - img: image, (C, H, W), numpy array.
        Returns:
            - img: image, (C, H, W), numpy array.
        '''
        img = (img + 1.0) / 2.0
        return img


class LMDBData(Dataset):
    def __init__(self, root, 
                 resolution=128, 
                 num_channels=1,
                 norm=True,
                 mean=0.0, std=5.0, id_list=None):
        super().__init__()
        self.root = root
        self.open_lmdb()
        self.resolution = resolution
        self.num_channels = num_channels
        self.norm = norm
        if id_list is None:
            self.length = self.txn.stat()['entries']
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.idx_map(idx)
        key = f'{idx}'.encode('utf-8')
        img_bytes = self.txn.get(key)
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(self.num_channels, self.resolution, self.resolution)
        if self.norm:
            img = self.normalize(img)
        return img
    
    def open_lmdb(self):
        self.env = lmdb.open(self.root, readonly=True, lock=False, create=False)
        self.txn = self.env.begin(write=False)

    def normalize(self, data):
        # By default, we normalize to zero mean and 0.5 std.
        return (data - self.mean) / (2 * self.std)
    
    def unnormalize(self, data):
        return data * 2 * self.std + self.mean


# this dataset only loads the single public test sample from GRMHD simulation
class BlackHole(Dataset):
    def __init__(self, root, resolution=64, original_resolution=400,
                 id_list=None):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.original_resolution = original_resolution

        if id_list is None:
            self.length = 1
            self.idx_map = lambda x: x
            self.id_list = list(range(self.length))
        else:
            id_list = parse_int_list(id_list)
            self.length = len(id_list)
            self.idx_map = lambda x: id_list[x]
            self.id_list = id_list

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        gt_path = os.path.join(self.root, 'blackhole_gt.pt')
        return torch.load(gt_path)


