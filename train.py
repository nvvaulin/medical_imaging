from config import ex
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from torchvision import models
import datasets
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from models import BasicClassifierModel
import os
import numpy as np
import re
import parse
import pandas as pd

# class JoinedLoader(DataLoader):
#     def __init__(self,*loaders):
#         self.loaders = loaders
#
#     @property
#     def batch_size(self):
#         return sum([i.bach_size for i in self.loaders])
#
#     def __len__(self):
#         return min([len(i) for i in self.loaders])
#
#     def __iter__(self):
#         return zip(*self.loaders)


@ex.capture
def load_train_val_test(dataset, batch_size=64,input_size=(224,224),num_workers=8):
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
                                            transforms.RandomSizedCrop(input_size,(0.8,1.2)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Scale(input_size),
                                            transforms.CenterCrop(input_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)
                                         ])

    train_val = datasets.__dict__[dataset['name']](mode='train',transform=train_transform,**dataset.get('params',{}))
    test = datasets.__dict__[dataset['name']](mode='test',transform=test_transform,**dataset.get('params',{}))
    assert (train_val.label_names!=test.label_names).sum()==0,'wrong!'
    train, val = random_split(train_val,[int(0.9*len(train_val)),len(train_val)-int(0.9*len(train_val))])
    train_loader = DataLoader(train, batch_size=batch_size,num_workers=num_workers,shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size,num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size,num_workers=num_workers)
    print('num_classes %d, dataset size: train %d; val %d; test %d'%(train_val.num_classes,len(train),len(val),len(test)))
    return train_val.label_names, train_loader,val_loader,test_loader

#
# @ex.capture
# def list_checkpoints(exp_root,exp_name,version):
#     checkpoints = glob.glob(ckpt
#
def load_backbone(name='resnet18',params={}):
    return models.__dict__[name](**params)
#
#
# def train
#
# @ex.capture
# def load_checkpoint(exp_root,exp_name,version,load_epoch='best'):
#     if load_epoch == 'best'
#
# @ex.command(model=None,load_epoch=1)
# def test

class ExistedModelCheckpoint(ModelCheckpoint):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._init_folder()

    def _init_folder(self):
        chpts = self.read_dir()

        if chpts is None or len(chpts) == 0:
            return

        if self.save_top_k:
            inx = np.argsort(chpts[self.monitor].values)
            if self.mode == 'max':
                inx = inx[::-1]
            inx = inx[:self.save_top_k]
            self.best_k_models = dict(zip(chpts['path'].values[inx],chpts[self.monitor].values[inx]))
            self.kth_best_model_path = chpts['path'].values[inx[-1]]

        if self.mode == 'min':
            self.best_model_score = chpts[self.monitor].min()
            self.best_model_path = chpts['path'].values[chpts[self.monitor].values.argmin()]
        else:
            self.best_model_score = chpts[self.monitor].max()
            self.best_model_path = chpts['path'].values[chpts[self.monitor].values.argmax()]
        self.last_model_path = chpts['path'].values[chpts['epoch'].values.argmax()]
        self.current_score = chpts['epoch'].values.max()

    def get_checkpoint_path(self,val='best',key='epoch'):
        if val is None:
            return None
        elif val == 'best':
            return self.best_model_path
        elif val == 'last':
            return self.last_model_path
        else:
            chpts = self.read_dir()
            if chpts is None:
                return None
            for i in range(len(chpts)):
                if chpts[key].values[i] == val:
                    return chpts['path'].values[i]
        return None

    def read_dir(self):
        if not os.path.exists(self.dirpath) or len(os.listdir(self.dirpath)) == 0:
            return None
        pattern = self.filename
        for i in re.findall('\{[A-Za-z_:\.0-9]+\}', pattern):
            pattern = pattern.replace(i, '%s=%s' % (re.findall('[A-Za-z_0-9]+', i)[0], i))
        chpts = [ dict(path=os.path.join(self.dirpath,i),**parse.search(pattern,i).named) for i in os.listdir(self.dirpath)]
        chpts = pd.DataFrame(chpts)
        chpts['epoch'] = pd.to_numeric(chpts['epoch'])
        return chpts




@ex.command
def test(exp_root,exp_name,version,_config,load_epoch=None):
    tb_logger = pl_loggers.TensorBoardLogger(exp_root, exp_name, version)
    label_names, _, _, test_loader = load_train_val_test(_config['dataset'])
    backbone = load_backbone(**_config.get('backbone', {}))
    model = BasicClassifierModel(backbone, label_names, _config['optimizer'], _config['scheduler'])
    checkpointer = ExistedModelCheckpoint(monitor='val_loss',
                                   mode='min',
                                   save_top_k=5,
                                   dirpath=os.path.join(exp_root, exp_name, version, 'checkpoints'),
                                   filename=_config['backbone']['name'] + '-{epoch}-{val_loss:.3f}-{train_loss:.3f}')
    trainer = pl.Trainer(logger=tb_logger,callbacks=[checkpointer],**_config.get('trainer',{}))
    trainer.test(model=model,test_dataloaders=test_loader,ckpt_path=checkpointer.get_checkpoint_path(load_epoch))

@ex.automain
def main(exp_root,exp_name,version,_config,load_epoch=None):
    tb_logger = pl_loggers.TensorBoardLogger(exp_root,exp_name,version)
    label_names,train_loader, val_loader, test_loader = load_train_val_test(_config['dataset'])
    backbone = load_backbone(**_config.get('backbone',{}))
    model = BasicClassifierModel(backbone,label_names,_config['optimizer'],_config['scheduler'])
    checkpointer = ExistedModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=5,
                                          dirpath = os.path.join(exp_root,exp_name,version,'checkpoints'),
                                          filename=_config['backbone']['name']+'-{epoch}-{val_loss:.3f}-{train_loss:.3f}')

    callbacks = [checkpointer,EarlyStopping(monitor='val_loss',patience=10)]
    trainer = pl.Trainer(logger=tb_logger,
                         resume_from_checkpoint=checkpointer.get_checkpoint_path(load_epoch),
                         callbacks=callbacks,**_config.get('trainer',{}))
    trainer.fit(model, train_loader, val_loader)
    print('load best epoch ',checkpointer.best_model_path)
    model.load_from_checkpoint(checkpointer.best_model_path)
    result = trainer.test(test_dataloaders=test_loader)
    print(result)
