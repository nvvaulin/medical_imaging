from torchvision import datasets
import pandas as pd
import numpy as np
import os
from .basic_dataset import BasicDataset


class CovidChestXRay(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/covid-chestxray-dataset',
                 mode='train',**kwargs):
        data = pd.read_csv(os.path.join(root,'metadata.csv'),sep=',')
        data = data[(data['finding'] != 'todo') & (data['finding'] != 'Unknown')]
        samples = data['filename'].values
        ignore_labels = set(['Aspergillosis', 'Aspiration', 'Chlamydophila', 'E.Coli', 
                             'H1N1', 'Herpes ', 'MRSA', 'Staphylococcus'])
        named_labels = np.array([[j for j in i.split('/') if not j in ignore_labels] for i in data['finding'].values])
        inx = np.array(open(os.path.join(root,mode.lower()+'.inx')).read().split('\n')).astype(np.int32)
        named_labels,samples = named_labels[inx],samples[inx]
        super().__init__(os.path.join(root,'images'),
                         samples,named_labels,multilabel=True,**kwargs)