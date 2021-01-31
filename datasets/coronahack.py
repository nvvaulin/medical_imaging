from torchvision import datasets
import pandas as pd
import numpy as np
import os
from .basic_dataset import BasicDataset



class Coronahack(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/coronahack',
                 mode='train',**kwargs):
        data = pd.read_csv(os.path.join(root,'Chest_xray_Corona_Metadata.csv'),sep=',')
        data = data[data['Dataset_type']==mode.upper()]
        samples = data['X_ray_image_name'].values
        named_labels = data['Label'].values
        super().__init__(os.path.join(root,'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'+mode.lower()),
                         samples,named_labels,**kwargs)