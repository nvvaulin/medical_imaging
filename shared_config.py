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
    exp_name='densenet_chest_covid_shared_nothing_bs16'
    version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    load_epoch='best'
    sampler = {'WeightedClassRandomSampler':{'names_weights': {'No Finding':0.1,'Pneumonia':0.1}}}
    exp_path = os.path.join(exp_root,exp_name,version)
    # optimizer = dict(SGD={'lr':0.01,'momentum':0.9,'weight_decay':1e-4})
    optimizer = dict(Adam={'lr':1e-3,'betas':(0.9, 0.999)})
    unfreeze_epoch = 4
    scheduler = dict(ReduceLROnPlateau={'factor':0.3,'patience':2,'min_lr':1e-4,'verbose':True})
    # scheduler = dict(MultiStepLR={'gamma':0.1,'milestones':[1000,  5000,10000]})
    dataset = ['ChestXRay','CovidChestXRay']
    val_dataset = 'CovidChestXRay'
    test_dataset = 'CovidChestXRay'
    # label_names = ['No Finding','COVID-19','Pneumonia']
    reduce_size = (256,256)
    batch_size = 16*2
    input_size = (224,224)
    shared_param_mask = 'nothing'

    trainer = dict(
        auto_select_gpus=True,
        gpus=1,
        max_epochs=15,
    )
    backbone = dict(densenet121={'num_classes':14})
    pretrained_backbone = 'converted_chexnet_model.pth.tar'
