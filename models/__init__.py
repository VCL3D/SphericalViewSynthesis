from .resnet360 import *

import sys

def get_model(name, model_params):
    if name == 'resnet_coord':
        return ResNet360(            
            # conv_type='standard', activation='elu', norm_type='none', \
            conv_type='coord', activation='elu', norm_type='none', \
            width=512,
        )
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)