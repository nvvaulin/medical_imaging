import pandas as pd
import numpy as np
import os
import glob
from .basic_dataset import BasicDataset,search


class ChestXRay(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/chestXray',**kwargs):
        data = pd.read_csv(os.path.join(root,'Data_Entry_2017.csv'),sep=',')
        samples = np.array([i[len(root)+1:] for i in glob.glob(os.path.join(root,'images*/images/*'))])
        samples = samples[search(np.array([i.split('/')[-1] for i in samples]),data['Image Index'].values)]
        named_labels = data['Finding Labels'].values
        named_labels = [i.split('|') for i in named_labels]
        super().__init__(root,samples,named_labels,**kwargs)


