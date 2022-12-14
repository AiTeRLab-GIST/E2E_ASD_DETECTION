import pdb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

class egemaps_dataset(Dataset):
    def __init__(self, df):
        self.fpath = df['path']
        self.label = df['asd']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        feat = np.load(self.fpath[index].replace('split_zeropadding','egemaps_zeropadding').replace('.wav','.npy'))
        label = self.label[index]
        return torch.from_numpy(feat).type(torch.float32), torch.tensor(label)
