from kernels import (
                    Linear,
                    Polynomial,
                    MaternOneHalf,
                    MaternFiveHalves,
                    MaternThreeHalves,
                    Gaussian,
                    Wendland
                    )
from numpy.polynomial import (chebyshev,
                              laguerre,
                              legendre,
                              hermite)
from quadratures import gLLNodesAndWeights as lobatto
import flax.linen as nn
import os, sys
import logging

def create_path(path, verbose=True):
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if verbose:
                print("Directory '%s' created successfully" % (path))
        #
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #

def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        if append:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        else:
            info_file_handler = logging.FileHandler(logpath, mode="w+")
        #
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()
    
    
#### for config 
##########################################################

kernel_mapping = {'Gaussian': Gaussian,
                  'Linear': Linear,
                  'Polynomial': Polynomial,
                  'MaternOneHalf': MaternOneHalf,
                  'MaternFiveHalves':MaternFiveHalves,
                  'MaternThreeHalves':MaternThreeHalves,
                  'Wendland':Wendland
                 }

non_linearity_mapping = {'gelu': nn.gelu,
                         'relu': nn.relu}

param_init_mapping = {'constant': nn.initializers.constant(1),
                      'he_normal': nn.initializers.he_normal()
                     }

quad_type_mapping = {'hermite':hermite.hermgauss,
                    'legendre':legendre.leggauss,
                     'chebyshev':chebyshev.chebgauss,
                     'laguerre':laguerre.laggauss,
                     'lobatto':lobatto
                    }
