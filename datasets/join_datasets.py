from torchvision import datasets
import numpy as np

class JoinDatasets(datasets.VisionDataset):
    def __init__(self,datasets,use_names=None):
        self.datasets = datasets
        self.labels,self.dinx,self.label_names = self._join_labels([i.labels for i in datasets],
                                                                   [i.label_names for i in datasets],use_names=use_names)
        self.offsets = [0]
        for i in self.dinx[:-1]:
            self.offsets.append(self.offsets[-1]+len(i))
            
    def __getitem__(self,item):
        for i,d,o in zip(self.dinx,self.datasets,self.offsets):
            if item < o+len(i):
                im,l = d[i[item-o]]
                return im,self.labels[item]
        assert False, 'index out of range %d'%item
    
    def __len__(self):
        return len(self.dinx[-1])+self.offsets[-1]
        
    @property
    def num_classes(self):
        return len(self.label_names)
    
    def _join_labels(self,labels,label_names,use_names=None):
        dinx = [np.arange(len(l)) for l in labels]
        if not use_names is None:
            use_names = set(use_names)
            masks = [np.array([i in use_names for i in j]) for j in label_names]
            labels = [l[:,m] for m,l in zip(masks,labels)]
            label_names = [n[m] for m,n in zip(masks,label_names)]
            dinx = [i[l.max(1) > .99] for i,l in zip(dinx,labels)]
            labels = [l[l.max(1) > .99] for l in labels]
        unames,inx = np.unique(np.concatenate(label_names),return_inverse=True)
        res = np.zeros((sum([len(i) for i in labels]),len(unames)),dtype=labels[0].dtype)+0.5
        i,j = 0,0        
        for l,n in zip(labels,label_names):
            res[i:i+len(l),inx[j:j+len(n)]] = l
            i+=len(l)
            j+=len(n)
        return res, dinx, unames