import pandas as pd
import numpy as np
import os
import glob
from .basic_dataset import BasicDataset

def search(a,key):
    inx = np.argsort(a)
    inx =  inx[np.searchsorted(a[inx],key)]
    assert (a[inx]!=key).sum()==0,'not all keys found'
    return inx


class ChestXRay(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/chestXray',mode='train',**kwargs):
        if mode.lower() == 'train':
            imlist = 'train_val_list.txt'
        elif mode.lower() == 'test':
            imlist = 'test_list.txt'
        else:
            assert False, 'unknown mode '+mode
        imlist = open(os.path.join(root,imlist)).read().split('\n')

        samples = np.array([i[len(root)+1:] for i in glob.glob(os.path.join(root,'images*/images/*'))])
        samples = samples[search(np.array([i.split('/')[-1] for i in samples]),imlist)]

        data = pd.read_csv(os.path.join(root,'Data_Entry_2017.csv'),sep=',')
        inx = search(data['Image Index'].values,imlist)
        named_labels = data['Finding Labels'].values[inx]
        named_labels = [i.split('|') for i in named_labels]
        super().__init__(root,samples,named_labels,multilabel=True,**kwargs)