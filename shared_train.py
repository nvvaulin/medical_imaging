from shared_config import ex
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import models
from utils.existed_checkpoint import ExistedModelCheckpoint
import datasets
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from utils import load_from_config,config_name
import torch
from utils import samplers
import numpy as np
from torch.utils.data import SequentialSampler,RandomSampler
from torch import nn
import torch
import re

def assign_parameter(model,name,param):
    names = name.split('.')
    for i in names[:-1]:
        model = model.__getattr__(i)
    model.__setattr__(names[-1],param)
    
class SharedModel(nn.Module):
    def __init__(self,model1,model2,shared_param_mask,**kwargs):
        super().__init__(**kwargs)
        share_params = []
        for n,p in model1.named_parameters():
            if re.match(shared_param_mask,n):
                share_params.append(n)
                assign_parameter(model2,n,p)
        self.model1 = model1
        self.model2 = model2
        
    def forward(self,x):
        if self.training:
            size = x.shape[0]//2
            x1,x2 = tuple(torch.split(x,[size,size]))
            return torch.cat([self.model1(x1),self.model2(x2)])
        else:
            return self.model2(x)


class MultyDatasetBatchSampler:
    def __init__(self,dataset,batch_size,shuffle=True,sampler=None,label_names=None):
        assert batch_size% len(dataset.datasets)==0,'wrong!'
        self.offsets = np.array(dataset.offsets)
        lo,hi = dataset.offsets,dataset.offsets[1:]+[None]
        if sampler is None:
            if shuffle:
                self.samplers = [RandomSampler(i) for i in dataset.datasets]
            else:
                self.samplers = [SequentialSampler(i) for i in dataset.datasets]
        else:
            self.samplers = [load_from_config(sampler, samplers)(labels=dataset.labels[l:h],
                                                                 label_names=dataset.label_names) for l,h in zip(lo,hi)]
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        for idx in zip(*self.samplers):
            batch.append(idx)
            if len(batch) == self.batch_size//len(self.samplers):
                batch = (np.array(batch)+self.offsets[None,:]).transpose().flatten()
                yield list(batch)
                batch = []

    def __len__(self) -> int:
        return min([len(i) for i in self.samplers])//(self.batch_size//len(self.samplers))
        
@ex.capture
def load_dataset(dataset, mode, sampler=None,batch_size=64,input_size=(224,224),num_workers=8,reduce_size=None,label_names=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = tuple(input_size)
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, (0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if not isinstance(dataset,list):
        dataset = [dataset]
    dataset = [load_from_config(i,datasets)(mode=mode,transform=transform,reduce_size=reduce_size) for i in dataset]
    dsizes = [len(i) for i in dataset]
    dataset = datasets.JoinDatasets(dataset,use_names=label_names)
    batch_sampler = MultyDatasetBatchSampler(dataset,batch_size,shuffle=(mode=='train'),sampler=sampler,label_names=label_names)
    loader = DataLoader(dataset,batch_sampler=batch_sampler, num_workers=num_workers)
    return loader,dataset






@ex.capture
def load_train_val_test(dataset=None,train_dataset=None,val_dataset=None,test_dataset=None,sampler=None,label_names=None):
    
    train = load_dataset(train_dataset or dataset,mode='train',sampler=sampler,label_names=label_names)
    if label_names is None:
        label_names = train[1].label_names
    val = load_dataset(val_dataset or dataset,mode='val',label_names=label_names,sampler=None)
    test_dataset = test_dataset or dataset
    if not isinstance(test_dataset,list):
        test_dataset = [test_dataset]
    test = [load_dataset(i,mode='test',label_names=label_names,sampler=None) for i in test_dataset]

    number_of_samples = np.concatenate([(i[1].labels==1).sum(0)[:,None] for i in [train,val,*test]],1)
    print('\n'.join(['{:15s} | '.format(n)+' | '.join(['{:6d}'.format(i) for i in c]) for n,c in zip(label_names,number_of_samples)]))
    return label_names, train[0],val[0],[i[0] for i in test]


def load_backdone(label_names,backbone,shared_param_mask,pretrained_backbone):
    backbone = load_from_config(backbone,models)()
    if not (pretrained_backbone is None):
        backbone.load_state_dict(torch.load(pretrained_backbone)['state_dict'],strict=True)
    
    if hasattr(backbone,'classifier'):
        backbone.classifier = nn.Sequential(nn.Linear(backbone.classifier.in_features, len(label_names)), nn.Sigmoid())
    else:
        backbone.fc= nn.Sequential(nn.Linear(self.backbone.fc.in_features, len(label_names)), nn.Sigmoid())
    return backbone
    
@ex.capture
def load_model(label_names,optimizer,scheduler,backbone,shared_param_mask,unfreeze_epoch=0,pretrained_backbone=None):
    backbone1 = load_backdone(label_names,backbone,shared_param_mask,pretrained_backbone)
    backbone2 = load_backdone(label_names,backbone,shared_param_mask,pretrained_backbone)        
    backbone = SharedModel(backbone1,backbone2,shared_param_mask)
    
    optimizer =  load_from_config(optimizer,torch.optim)
    lr_scheduler = load_from_config(scheduler,torch.optim.lr_scheduler)
    model = models.BasicClassifierModel(backbone, label_names, optimizer, lr_scheduler,unfreeze_epoch=unfreeze_epoch)
    return model

@ex.capture
def load_trainer(exp_root,exp_name,version,_config,load_epoch=None):
    tb_logger = pl_loggers.TensorBoardLogger(exp_root,exp_name,version)
    checkpointer = ExistedModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=5,
                                          dirpath = os.path.join(exp_root,exp_name,version,'checkpoints'),
                                          filename=config_name(_config['backbone'])+'-{epoch}-{val_loss:.3f}-{train_loss:.3f}')

    callbacks = [checkpointer,EarlyStopping(monitor='val_loss',patience=10)]
    trainer = pl.Trainer(logger=tb_logger,
                         resume_from_checkpoint=checkpointer.get_checkpoint_path(load_epoch),
                         callbacks=callbacks,**_config.get('trainer',{}))
    return trainer,checkpointer

@ex.capture
def write_results(path,results,exp_root,exp_name,version):
    open(os.path.join(exp_root,exp_name,version,'%s.csv'%(path.split('/')[-1])),'a').write('\n'.join(['%s,%s'%(k,str(v)) for k,v in results[0].items()])+'\n')

@ex.command
def test(load_epoch):
    label_names, _, _, test_loaders = load_train_val_test()
    model = load_model(label_names)
    trainer,checkpointer = load_trainer()
    for test_loader in test_loaders:
        results = trainer.test(model=model,test_dataloaders=test_loader)
        write_results(checkpointer.get_checkpoint_path(load_epoch),results)


@ex.automain
def main(load_epoch):
    label_names,train_loader, val_loader, test_loaders = load_train_val_test()
    model=load_model(label_names)
    trainer,checkpointer = load_trainer()
    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        if not checkpointer.get_checkpoint_path(load_epoch) is None:
            for test_loader in test_loaders:
                results = trainer.test(model=model, test_dataloaders=test_loader)
                write_results(checkpointer.get_checkpoint_path(load_epoch), results)