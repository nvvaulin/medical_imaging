import numpy as np
import os
from .basic_dataset import BasicDataset


class CovidAID(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/covidaid',mode='train',**kwargs):
        lst = np.array([i.split('\t') for i in open(os.path.join(root,mode+'.txt'),'r').read().split('\n') if len(i) > 1])
        super().__init__(root,lst[:,0],lst[:,1:],mode=None,**kwargs)
