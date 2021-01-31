from sacred import Experiment
from sacred.commands import save_config,_format_config
import os
import glob
import datetime
import shutil
import logging

ex = Experiment('default')

def mkdir(fpath):
    dpath = os.path.dirname(fpath)
    if not os.path.exists(dpath):
        os.makedirs(dpath)


@ex.pre_run_hook
def set_up_loging(exp_path,_config,_run,loglevel='INFO'):
    spath = os.path.join(exp_path,'scources')
    lpath = os.path.join(exp_path,'log.txt')
    cpath = os.path.join(exp_path,'config.json')


    for src in (glob.glob('./*.py')+glob.glob('./*/*.py')):
        dst = os.path.join(spath,src[2:])
        mkdir(dst)
        shutil.copy(src,dst)

    mkdir(lpath)
    handler = logging.FileHandler(lpath)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s',
                                           datefmt='%m-%d %H:%M:%S'))
    _run.run_logger.setLevel(loglevel)
    _run.run_logger.addHandler(handler)

    mkdir(cpath)
    save_config(_run.config,_run.run_logger,cpath)
    _run.run_logger.info(_format_config(_run.config,_run.config_modifications))


@ex.config
def config():
    exp_root='exps'
    exp_name='multilabel'
    version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(exp_root,exp_name,version)
    optimizer = dict(name='SGD',params={'lr':0.01,'momentum':0.9,'weight_decay':1e-4})

    scheduler = dict(name='ReduceLROnPlateau',params={'factor':0.3,'patience':5,'min_lr':1e-5,'verbose':True})
    # scheduler = dict(name='MultiStepLR',params={'gamma':0.1,'milestones':[]})
    dataset = dict(name='ChestXRay',params={'reduce_size':(256,256)})#'Coronahack')
    batch_size = 16
    input_size = (224,224)

    trainer = dict(
        auto_select_gpus=True,
        gpus=1,
        max_epochs=100,
    )
    backbone = dict(name = 'densenet121',params={'pretrained':True})


# def load_config(exp_name,config_path='config.yml',exp_root='exps',loglevel='INFO'):
#     import yaml
#     from pytorch_lightning import loggers as pl_loggers
#     name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     exp_root = os.path.join(exp_root, exp_name,name)
#     config = yaml.load(open(config_path))
#
#     lpath = os.path.join(exp_root, 'log.txt')
#     cpath = os.path.join(exp_root, 'config.yml')
#
#     for src in (glob.glob('./*.py') + glob.glob('./*/*.py')):
#         dst = os.path.join(exp_root,'scources', src[2:])
#         mkdir(dst)
#         shutil.copy(src, dst)
#
#     mkdir(lpath)
#     handler = logging.FileHandler(lpath)
#     handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s',
#                                            datefmt='%m-%d %H:%M:%S'))
#     _run.run_logger.setLevel(loglevel)
#     _run.run_logger.addHandler(handler)
#
#     mkdir(cpath)
#     yaml.dump(config,open(cpath,'w'))
#     tb_logger = pl_loggers.TensorBoardLogger(exp_root)
#     return tb_logger
#
# def build_object(_config,_module,**kwargs):
#     if isinstance(config,str):
#         return module.__dict__[config]
#     elif isinstance(config,dict):
#         assert len(config) == 1,'wrong'
#         return module.__dict__[list(config.keys())[0]](**list(config.values())[0])
#     else:
#         assert False,'config must be str or dict, provided '+str(type(config))