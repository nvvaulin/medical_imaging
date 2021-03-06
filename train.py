from config import ex
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
    dataset = datasets.JoinDatasets(dataset,use_names=label_names)
    if not (sampler is None):
        sampler = load_from_config(sampler, samplers)(labels=dataset.labels,
                                                      label_names=dataset.label_names)
    if mode=='train' and (sampler is None):
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=sampler)

    return loader,dataset






@ex.capture
def load_train_val_test(dataset=None,train_dataset=None,val_dataset=None,test_dataset=None,sampler=None,
                        batch_size=64, input_size=(224, 224), num_workers=8, label_names=None,reduce_size=None):
    train = load_dataset(train_dataset or dataset,mode='train',sampler=sampler,batch_size=batch_size,input_size=input_size,num_workers=num_workers,label_names=label_names,reduce_size=reduce_size)
    if label_names is None:
        label_names = train[1].label_names
    val = load_dataset(val_dataset or dataset,mode='val',batch_size=batch_size,input_size=input_size,num_workers=num_workers,label_names=label_names,reduce_size=reduce_size)
    test_dataset = test_dataset or dataset
    if not isinstance(test_dataset,list):
        test_dataset = [test_dataset]
    test = [load_dataset(i,mode='test',batch_size=batch_size,input_size=input_size,num_workers=num_workers,label_names=label_names,reduce_size=reduce_size) for i in test_dataset]

    number_of_samples = np.concatenate([(i[1].labels==1).sum(0)[:,None] for i in [train,val,*test]],1)
    print('\n'.join(['{:15s} | '.format(n)+' | '.join(['{:6d}'.format(i) for i in c]) for n,c in zip(label_names,number_of_samples)]))
    return label_names, train[0],val[0],[i[0] for i in test]



@ex.capture
def load_model(label_names,optimizer,scheduler,backbone,unfreeze_epoch=0,pretrained_backbone=None):
    backbone = load_from_config(backbone,models)()
    if not (pretrained_backbone is None):
        backbone.load_state_dict(torch.load(pretrained_backbone)['state_dict'],strict=True)
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