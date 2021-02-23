from pytorch_lightning.callbacks import ModelCheckpoint
import os
import re
import parse
import pandas as pd
import numpy as np


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
            return self.best_model_path if len(self.best_model_path) > 0 else None
        elif val == 'last':
            return self.last_model_path if len(self.last_model_path) > 0 else None
        else:
            chpts = self.read_dir()
            if chpts is None:
                return None
            for i in range(len(chpts)):
                if chpts[key].values[i] == val:
                    return chpts['path'].values[i]
        return None

    def read_dir(self):
        if (not os.path.exists(self.dirpath)) or (len(os.listdir(self.dirpath)) == 0):
            return None
        pattern = self.filename
        for i in re.findall('\{[A-Za-z_:\.0-9]+\}', pattern):
            pattern = pattern.replace(i, '%s=%s' % (re.findall('[A-Za-z_0-9]+', i)[0], i))
        chpts = [ dict(path=os.path.join(self.dirpath,i),**parse.search(pattern,i).named) for i in os.listdir(self.dirpath)]
        chpts = pd.DataFrame(chpts)
        chpts['epoch'] = pd.to_numeric(chpts['epoch'])
        return chpts

