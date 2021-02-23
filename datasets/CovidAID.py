from torchvision import datasets
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm
import torch

class CovidAID(datasets.VisionDataset):
    def __init__(self,root,mode,):