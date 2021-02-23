from torchvision import datasets
import pandas as pd
import numpy as np
import os
from .basic_dataset import BasicDataset


class Coronahack(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/coronahack',
                 mode='train',**kwargs):
        data = pd.read_csv(os.path.join(root,'Chest_xray_Corona_Metadata.csv'),sep=',')
        data.fillna('unknown', inplace=True)
        data['Label'][data['Label']=='Normal'] = 'No Finding'
        data['Label'][data['Label']=='Pnemonia'] = 'Pneumonia'
        data['Label_1_Virus_category'][data['Label_1_Virus_category']=='Virus'] = 'Viral'
        data['Label_1_Virus_category'][data['Label_1_Virus_category']=='bacteria'] = 'Bacterial'

        samples = data['X_ray_image_name'].values
        named_labels = np.array([[j for j in i if j != 'unknown'] for i in zip(data['Label'].values,
                                                                      data['Label_1_Virus_category'].values,
                                                                      data['Label_2_Virus_category'].values)])
        mode_mask =  data['Dataset_type']==mode.upper()
        super().__init__(os.path.join(root,'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'+mode.lower()),
                         samples[mode_mask],named_labels[mode_mask],multilabel=True,label_names=np.unique(np.array([i for j in named_labels for i in j])),**kwargs)