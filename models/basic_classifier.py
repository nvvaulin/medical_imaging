import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import auroc,accuracy
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt

class BasicClassifierModel(pl.LightningModule):
    def __init__(self, backbone,class_names,
                 optimizer={'name':'Adam','params':{'lr':0.1}},
                 scheduler={'name':'MultiStepLR','params':{'gamma':0.1,'milestones':[100]}}):
        super().__init__()
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.backbone = backbone
        self.num_classes = len(class_names)

        if hasattr(self.backbone,'classifier'):
            self.backbone.classifier = nn.Sequential(nn.Linear(self.backbone.classifier.in_features, self.num_classes), nn.Sigmoid())
        else:
            self.backbone.fc= nn.Sequential(nn.Linear(self.backbone.fc.in_features, self.num_classes), nn.Sigmoid())
        self.loss = nn.BCELoss()
        self.class_names = class_names

    def forward(self, x):
        embedding = self.backbone(x)
        pred = self.fc(embedding)
        return pred

    def _step(self,type,batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y)
        self.log('%s_loss'%type, loss)
        return {'loss':loss,'target':y,'pred':y_hat.detach()}

    def _epoch_end(self,type,outputs):
        pred = np.concatenate([i['pred'].cpu().numpy() for i in outputs],0)
        target = np.concatenate([i['target'].cpu().numpy() for i in outputs],0)
        for i,n in enumerate(self.class_names):
            n = n.replace(' ','_')
            if len(np.unique(target[:,i])) == 1:
                continue
            fpr,tpr,_ = roc_curve(target[:,i],pred[:,i])
            fig = plt.figure(figsize=(10,10))
            roc_auc = roc_auc_score(target[:,i],pred[:,i])
            self.log('%s_%s_auc'%(type,n), roc_auc)
            plt.plot(fpr,tpr,label='ep-%d;roc_auc-%.2f'%(self.current_epoch,roc_auc))
            self.logger.experiment.add_figure('%s_%s_roc%d'%(type,n,self.current_epoch),fig)
            print('%s %s auc: %.3f'%(type,n,roc_auc))


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
        optimizer =  torch.optim.__dict__[self.optimizer['name']](self.parameters(), **self.optimizer['params'])
        lr_scheduler = torch.optim.lr_scheduler.__dict__[self.scheduler['name']](optimizer,**self.scheduler['params'])
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}