def place_kwargs_decorator(func,verbose,kwargs):
    def f(*args,**new_kwargs):
        if verbose:
            print('call %s(%s,%s)'%(func.__name__,
                                    ','.join([str(i) for i in args]),
                                    ','.join(['%s=%s'%(k,str(v)) for k,v in kwargs.items()])))
        _kwargs = dict(**kwargs)
        _kwargs.update(new_kwargs)
        return func(*args,**_kwargs)
    return f

def load_from_config(config,module,verbose=False):
    if isinstance(config,str):
        name=config
        kwargs={}
    elif isinstance(config,dict):
        assert len(config)==1
        for name,kwargs in config.items():
            assert isinstance(kwargs,dict)
    else:
        return config
    return place_kwargs_decorator(module.__dict__[name],verbose,kwargs)


def config_name(config):
    if isinstance(config,str):
        name=config
        kwargs={}
    elif isinstance(config,dict):
        assert len(config)==1
        for name,kwargs in config.items():
            assert isinstance(kwargs,dict)
    else:
        name = config.__name__
    return name