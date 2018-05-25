import importlib

def load_support(context, obj):
    module = obj.__class__.__module__.split('.')[0]
    try:
        mod = importlib.import_module(module, 'numba.support')
    except ImportError:
        raise NotImplementedError("No support for {} found. Try pip install numba-support-{}.".format(module))

    mod.init(context)