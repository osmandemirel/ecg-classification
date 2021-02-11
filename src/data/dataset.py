import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import utils
from pathlib import Path
import glob
from random import shuffle
import random
from scipy import io


class ECGDataset(Dataset):
    def __init__(self, root_dir, class_list, seed=1773):
        """
        :param root_dir: get train or test image list using glob
            then split them to train and validation using train_val_split
            PS: FOLDERS BASED ON LABEL NAMES
        :param transform: transforms to be applied on data set
        """
        self.class_list = class_list
        self.labels = self._paths_to_labels(root_dir)
        self.file_list = self._load_data(root_dir)
        self.dataset = self._shuffle(seed)

    def _get_labels(self, path):
        with open(path, 'r') as f:
            header = f.readlines()

        starts = [header.index(l) for l in header if l.startswith("#Dx")]
        label = header[starts[0]].strip().split(' ')[1].split(',')
        class_tensor = torch.zeros(len(self.class_list))
        class_tensor[[self.class_list.index(eval(l)) for l in label if eval(l) in self.class_list]] = 1
        return class_tensor

    def _paths_to_labels(self, path):
        glob_path = path + os.sep + '*.hea'
        label_paths = glob.glob(pathname=glob_path, recursive=True)
        labels = [self._get_labels(path) for path in label_paths]
        return labels

    def _load_data(self,path):
        """
        :param path: path to dataset. folder should be split as /covid, /normal
        :return: list of tuples for path path and label.
            :return image_path: path to read path
            :return label: 1 for covid, other for normal patient
        """
        glob_path = path + os.sep + '*.mat'
        paths = glob.glob(pathname=glob_path, recursive=True)
        assert len(paths)==len(self.labels)
        return [(paths[idx], self.labels[idx]) for idx in range(len(self.labels))]

    def _shuffle(self, seed):
        """
        :param split: split ratio
        :param seed: seed for splitting train and validation
        :return: splitted train and validation sets
        """
        dataset = self.file_list.copy()
        random.Random(seed).shuffle(dataset)
        return dataset

    def __len__(self):
        """
        :return: returns length of dataset based on train flag
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        :param idx: next image index
        :return: return next image in line with its label
        """
        file_path, label = self.dataset[idx]
        x = io.loadmat(file_path)
        recording = torch.tensor(x['val'])

        return (recording, label)
