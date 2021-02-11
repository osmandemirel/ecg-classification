import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils
from pathlib import Path
import glob
from random import shuffle
import random
from torchvision.datasets import ImageFolder
import cv2
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self,root_dir,
                train=True,
                transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                ]),
                train_val_split=0.8,
                seed=2021
                ):
        """
        :param root_dir: get train or test image list using glob
            then split them to train and validation using train_val_split
            PS: FOLDERS BASED ON LABEL NAMES
        :param transform: transforms to be applied on data set
        """
        self.idx_labels = self._paths_to_labels(root_dir)
        self.train = train
        self.transform = transform
        self.image_list = self._load_data(root_dir)
        self.train_set,self.val_set = self._train_val_split(train_val_split,seed)

    def _paths_to_labels(self, path):
        glob_path = path + os.sep + '*' + os.sep + '*'
        paths = glob.glob(pathname=glob_path, recursive=True)
        labels = [path.split(os.sep)[-2] for path in paths]
        unique_labels = sorted(set(labels))
        return {label:idx for idx,label in enumerate(unique_labels)}

    def _load_data(self,path):
        """
        :param path: path to dataset. folder should be split as /covid, /normal
        :return: list of tuples for path path and label.
            :return image_path: path to read path
            :return label: 1 for covid, other for normal patient
        """
        glob_path = path + os.sep + '*' + os.sep + '*'
        paths = glob.glob(pathname=glob_path, recursive=True)
        label_names = [path.split(os.sep)[-2] for path in paths]
        labels = [self.idx_labels[label] for label in label_names]

        assert len(paths)==len(labels)

        return [(paths[idx],labels[idx]) for idx in range(len(labels))]

    def _train_val_split(self, split,seed):
        """
        :param split: split ratio
        :param seed: seed for splitting train and validation
        :return: splitted train and validation sets
        """
        dataset = self.image_list.copy()
        random.Random(seed).shuffle(dataset)
        train_set = dataset[:int(len(dataset)*split)]
        val_set = dataset[int(len(dataset)*split):]
        return train_set,val_set

    def __len__(self):
        """
        :return: returns length of dataset based on train flag
        """
        if self.train:
            return len(self.train_set)
        else:
            return len(self.val_set)

    def __getitem__(self, idx):
        """
        :param idx: next image index
        :return: return next image in line with its label
        """
        img_path,label = self.train_set[idx] if self.train else self.val_set[idx]
        """
        if self.train:
            img_path,label = self.train_set[idx]
        else:
            img_path, label = self.val_set[idx]
        """
        img = Image.open(img_path)
        img = self.transform(img)

        return (img, label)
