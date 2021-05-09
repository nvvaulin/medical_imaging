from torchvision import datasets
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm

def search(a,key):
    inx = np.argsort(a)
    inx =  inx[np.searchsorted(a[inx],key)]
    assert (a[inx]!=key).sum()==0,'not all keys found'
    return inx

class BasicDataset(datasets.VisionDataset):
    def __init__(self,root,samples,named_labels,mode, transforms=None, transform=None,target_transform=None,reduce_size=None):
        self.label_names, labels = self._parse_multilabel(named_labels)
        split = self.get_split(samples,root,mode)
        self.samples,self.labels = samples[split],labels[split]

        self.loader = datasets.folder.default_loader
        if not (reduce_size is None):
            reduce_size = tuple(reduce_size)
            out_prefix = os.path.join(root,'resized_%d_%d'%reduce_size)
            if not os.path.exists(os.path.join(out_prefix,self.samples[-1])):
                self._reduce_size(root,reduce_size)
            root = out_prefix
        super().__init__(root,transforms, transform,target_transform)

    def get_split(self,samples,root,mode):
        if mode is None:
            return np.arange(len(samples))
        imlist = open(os.path.join(root,mode+'.lst')).read().split('\n')
        inx = search(samples,imlist)
        return inx

    def _parse_multilabel(self,named_labels):
        named_labels = [np.array(i) for i in named_labels]
        label_names = np.unique(np.concatenate(named_labels))
        labels = []
        for i,nl in enumerate(named_labels):
            l = np.zeros(len(label_names),dtype=np.float32)
            if len(nl) > 0:
                l[np.searchsorted(label_names,nl)] = 1.
            labels.append(l)
        return label_names,np.array(labels)

    def __getitem__(self, item):
        sample = os.path.join(self.root,self.samples[item])
        sample = self.loader(sample)
        label = self.labels[item]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return sample, label



    @property
    def num_classes(self):
        return len(self.label_names)

    def __len__(self):
        return len(self.labels)

    def _reduce_size(self,root,imsize):
        print('reduce size of dataset%s'%root)
        out_prefix = os.path.join(root,'resized_%d_%d'%imsize)
        for i in tqdm(self.samples):
            src = os.path.join(root,i)
            dst = os.path.join(out_prefix,i)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            im = cv2.imread(src)
            if im.shape[0]>imsize[1] or im.shape[1]>imsize[0]:
                im = cv2.resize(im,imsize)
                cv2.imwrite(dst,im)
            else:
                shutil.copy(src,dst)
