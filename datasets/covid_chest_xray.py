import pandas as pd
import numpy as np
import os
from .basic_dataset import BasicDataset


class CovidChestXRay(BasicDataset):
    def __init__(self,root='/media/nvme/data/datasets/medical_datasets/covid-chestxray-dataset',**kwargs):
        data = pd.read_csv(os.path.join(root,'metadata.csv'),sep=',')
        samples = data['filename'].values
        label_names = ['Bacterial', 'COVID-19', 'Fungal', 'Klebsiella', 'Legionella', 'Lipoid',
                       'No Finding', 'Pneumocystis', 'Pneumonia', 'SARS', 'Streptococcus',
                       'Tuberculosis', 'Viral']
        named_labels = np.array([[j for j in i.split('/') if j in label_names] for i in data['finding'].values])
        super().__init__(os.path.join(root,'images'), samples,named_labels,**kwargs)