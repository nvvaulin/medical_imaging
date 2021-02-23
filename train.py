from config import ex
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torchvision import models
from utils.existed_checkpoint import ExistedModelCheckpoint
import datasets
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from torch.utils.data import DataLoader
from models import BasicClassifierModel
from datasets import train_val_split
import os
import numpy as np
from utils import load_from_config,config_name
import torch
from utils import samplers

@ex.capture
def load_train_val_test(dataset, batch_size=64,input_size=(224,224),num_workers=8,sampler=None,use_names=None):
    # imagenet maan and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = tuple(input_size)
    test_transform = transforms.Compose([
                                    transforms.Resize(input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)
                                   ])

    train_transform = transforms.Compose([
                                            transforms.RandomResizedCrop(input_size,(0.8,1.2)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)
                                         ])
    train_val = [load_from_config(i,datasets)(mode='train',transform=train_transform) for i in dataset]
    test      = [load_from_config(i,datasets)(mode='test' ,transform=test_transform ) for i in dataset]
    train_val = datasets.JoinDatasets(train_val,use_names=use_names)
    test      = datasets.JoinDatasets(test,use_names=use_names)
    assert (train_val.label_names!=test.label_names).sum()==0,'wrong!'
    train, val = train_val_split(train_val,0.8)
    if sampler is None:
        train_loader = DataLoader(train, batch_size=batch_size,num_workers=num_workers,shuffle=True)
    else:
        train_labels = train.dataset.labels[train.indices]==1
        sampler = load_from_config(sampler, samplers)(labels=train_labels,
                                                      label_names=train_val.label_names)
        train_labels = train_labels[np.array(list(iter(sampler)))]
        print('re-weighting classes:\n' +
              '\n'.join(['{:15s}| {:6d}({:2.2f}%)'.format(*i) for i in
                         zip(train_val.label_names,
                             (train_labels == 1).sum(0),
                             (train_labels== 1).mean(0) * 100)]))
        train_loader = DataLoader(train, batch_size=batch_size,num_workers=num_workers,sampler=sampler)

    val_loader = DataLoader(val, batch_size=batch_size,num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size,num_workers=num_workers)
    print('num_classes %d, dataset size: train %d; val %d; test %d'%(train_val.num_classes,len(train),len(val),len(test)))
    return train_val.label_names, train_loader,val_loader,test_loader



@ex.capture
def load_model(label_names,optimizer,scheduler,backbone,unfreeze_epoch=0,pretrained_backbone=None):
    backbone = load_from_config(backbone,models)()
    if not (pretrained_backbone is None):
        import re
        params = torch.load(pretrained_backbone)['state_dict']
        mapped_params = {}
        #map names
        for k, v in params.items():
            k = k[len('module.densenet121.'):]
            if re.match('[0-9]+',k.split('.')[-2]):
                k = k.split('.')
                k = '.'.join(k[:-3]+[k[-3]+k[-2],k[-1]])
            mapped_params[k] = v
        backbone.load_state_dict(mapped_params,strict=False)
    optimizer =  load_from_config(optimizer,torch.optim)
    lr_scheduler = load_from_config(scheduler,torch.optim.lr_scheduler)
    model = BasicClassifierModel(backbone, label_names, optimizer, lr_scheduler,unfreeze_epoch=unfreeze_epoch)
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

@ex.command
def train_fc(_config):
    label_names,train_loader, val_loader, test_loader = load_train_val_test(_config['dataset'])
    model=load_model(label_names)
    print('freeze model')
    for param in model.features.parameters():
        param.requires_grad = False
    trainer,checkpointer = load_trainer()
    trainer.fit(model, train_loader, val_loader)
    _ = trainer.test(test_dataloaders=test_loader)
    print(checkpointer.get_checkpoint_path('best'))

@ex.command
def test(_config):
    label_names, _, _, test_loader = load_train_val_test(_config['dataset'])
    model = load_model(label_names)
    trainer,_ = load_trainer()
    trainer.test(model=model,test_dataloaders=test_loader)

@ex.automain
def main(_config):
    label_names,train_loader, val_loader, test_loader = load_train_val_test(_config['dataset'])
    model=load_model(label_names)
    trainer,_ = load_trainer()
    trainer.fit(model, train_loader, val_loader)
    _ = trainer.test(test_dataloaders=test_loader)
