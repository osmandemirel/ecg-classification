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
    def __init__(self, root_dir, seed=1773):
        """
        :param root_dir: get train or test image list using glob
            PS: FOLDERS BASED ON LABEL NAMES
        :param transform: transforms to be applied on data set
        """
        self.labels = None
        self.file_list = self._load_data(root_dir)
        self.dataset = self._shuffle(seed)

    def _get_label(self, path):
        """
        There are over 100 labels in data but we only interest in 3 of them
        which is 
        "sinus_rhythm" with Dx code 426783006, 
        "atrial_fibrillation" with Dx code 164889003,
        "other" with remaining Dx codes.
        """
        with open(path, 'r') as f:
            header = f.readlines()

        starts = [header.index(l) for l in header if l.startswith("#Dx")]
        label = header[starts[0]].strip().split(' ')[1].split(',')
        if "426783006" in label:
            # sinus rhythm
            label = 0
        elif "164889003" in label:
            # atrial fibrillation
            label = 1
        else:
            # other
            label = 2
        return label

    def _load_data(self,path):
        """
        :param path: path to dataset. folder should be split as /covid, /normal
        :return: list of tuples for path path and label.
            :return path: path to read path
            :return record path and label tuple: 0 for sinus, 1 for af and 2 for other
        """
        glob_path = path + os.sep + '*.mat'
        paths = glob.glob(pathname=glob_path, recursive=True)
        self.labels = [self._get_label(p.replace(".mat", ".hea")) for p in paths]
        assert len(paths) == len(self.labels)
        return [(paths[idx], self.labels[idx]) for idx in range(len(self.labels))]

    def _shuffle(self, seed):
        """
        :param seed: seed for shuffling dataset
        :return: shuffled train and validation sets
        """
        dataset = self.file_list.copy()
        random.Random(seed).shuffle(dataset)
        return dataset

    def __len__(self):
        """
        :return: returns length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        :param idx: next record index
        :return: return next record in line with its label
        """
        file_path, label = self.dataset[idx]
        x = io.loadmat(file_path)
        recording = torch.tensor(x['val'], dtype=torch.float)
        norm = torch.norm(recording,2,1,True)
        recording = torch.div(recording, norm)
        recording[torch.isnan(recording)] = 0 
        recording.resize_([12, 5000])

        return (recording, label)
