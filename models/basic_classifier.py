import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import re


class BasicClassifierModel(pl.LightningModule):
    def __init__(self, backbone,class_names,
                 optimizer={'Adam':{'lr':0.1}},
                 scheduler={'MultiStepLR':{'gamma':0.1,'milestones':[100]}},
                 unfreeze_epoch=0):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.backbone = backbone
        self.num_classes = len(class_names)

#         if hasattr(self.backbone,'classifier'):
#             self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier.in_features, self.num_classes), nn.Sigmoid())
#         else:
#             self.backbone.fc= nn.Sequential(nn.Linear(self.backbone.fc.in_features, self.num_classes), nn.Sigmoid())
        self.loss = nn.BCELoss(reduction='none')
        self.class_names = class_names

    def forward(self, x):
        embedding = self.backbone(x)
        pred = self.fc(embedding)
        return pred

    def _step(self,type,batch, batch_idx):
        x, y = batch

        y_hat = self.backbone(x)
        mask = (torch.abs(y-0.5)>0.1)
        loss = (self.loss(y_hat, y)*mask).sum()/mask.sum()
        self.log('%s_loss'%type, loss)
        return {'loss':loss,'target':y,'pred':y_hat.detach()}

    def freeze_mask(self,mask=''):
        names = []
        for name,param in self.named_parameters():
            if re.match(mask,name):
                param.requires_grad = True
                names.append(name)
        print('freeze paremeters: %s'%','.join(names))

    def unfreeze_mask(self, mask=''):
        names = []
        for name,param in self.named_parameters():
            if re.match(mask,name):
                param.requires_grad = True
                names.append(name)
        print('unfreeze paremeters: %s'%','.join(names))

    def on_epoch_start(self):
        if self.current_epoch == 0:
            print('freeze model')
            self.freeze()
            self.train(True)
            self.unfreeze_mask('.*classifier.*')
        if self.current_epoch == self.unfreeze_epoch:
            print('unfreeze model')
            self.unfreeze()  # Or partially unfreeze
            self.freeze_mask('.*conv.*weight')

    def _epoch_end(self,type,outputs):
        pred = np.concatenate([i['pred'].cpu().numpy() for i in outputs],0)
        target = np.concatenate([i['target'].cpu().numpy() for i in outputs],0)
        avg_auc = []
        avg_tpr = []
        for i,n in enumerate(self.class_names):
            n = n.replace(' ','_')
            mask = np.abs(target[:,i]-0.5)>0.1
            t = target[:,i][mask]
            p = pred[:,i][mask]
            if len(np.unique(t)) < 2:
                print('skip %s'%n,len(t),(t==0).sum(),(t==1).sum(),(t==0.5).sum())
                continue
            fpr,tpr,_ = roc_curve(t,p)
            fig = plt.figure(figsize=(10,10))
            roc_auc = roc_auc_score(t,p)
            avg_auc.append(roc_auc)
            avg_tpr.append(tpr[np.searchsorted(fpr,1e-3)])
            self.log('%s_%s_auc'%(type,n), roc_auc)
            self.log('%s_%s_tpr'%(type,n), tpr[np.searchsorted(fpr,1e-3)])
            plt.semilogx(fpr,tpr,label='ep-%d;roc_auc-%.2f'%(self.current_epoch,roc_auc))
            self.logger.experiment.add_figure('%s_%s_roc%d'%(type,n,self.current_epoch),fig)
            print('%s %s auc: %.3f;  tpr@1e-3: %.3f'%(type,n,roc_auc,tpr[np.searchsorted(fpr,1e-3)]))

        n = 'avg'
        avg_auc = np.array(avg_auc).mean()
        avg_tpr = np.array(avg_tpr).mean()
        print('%s %s auc: %.3f;  tpr@1e-3: %.3f'%(type,n,avg_auc,avg_tpr))
        self.log('%s_%s_auc'%(type,n), avg_auc  )
        self.log('%s_%s_tpr'%(type,n), avg_tpr)


    def training_step(self, batch, batch_idx):
        return self._step('train',batch,batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step('val',batch,batch_idx)

    def validation_epoch_end(self,outputs):
        self._epoch_end('val',outputs)

    def train_epoch_end(self,outputs):
        self._epoch_end('train',outputs)

    def test_epoch_end(self,outputs):
        self._epoch_end('test',outputs)

    def test_step(self, batch, batch_idx):
        return self._step('test',batch,batch_idx)

    def configure_optimizers(self):
        optimizer =  self.optimizer(self.parameters())
        lr_scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}