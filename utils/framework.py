import torch
import datetime
import numpy
import random 

from .opt import *
from .visualization import *

def initialize(args): #TODO: add visdom count as argument       
    # create and init device
    print("{} | Torch Version: {}".format(datetime.datetime.now(), torch.__version__))
    if args.seed > 0:
        print("Set to reproducibility mode with seed: {}".format(args.seed))    
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        numpy.random.seed(args.seed)        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   
        random.seed(args.seed)        
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device("cuda:{}" .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu")
    print("Training {0} for {1} epochs using a batch size of {2} on {3}".format(args.name, args.epochs, args.batch_size, device))
    # create visualizer
    visualizer = (NullVisualizer(), NullVisualizer())\
        if args.visdom is None\
        else (
            VisdomPlotVisualizer(args.name + "_plots_", args.visdom),
            VisdomImageVisualizer(args.name + "_images_", args.visdom,\
                count=2 if 2 <= args.batch_size else args.batch_size)
        )
    if args.visdom is None:
        args.visdom_iters = 0
    # create & init model
    model_params = {
        'width': 512,
        'height': 256,
        'configuration': args.configuration,
    }
    return device, visualizer, model_params

def init_optimizer(model, args):
    opt_params = OptimizerParameters(learning_rate=args.lr, momentum=args.momentum,\
        momentum2=args.momentum2, epsilon=args.epsilon)
    optimizer = get_optimizer(args.optimizer, model.parameters(), opt_params)
    if args.opt_state is not None:
        opt_state = torch.load(args.opt_state)
        print("Loading previously saved optimizer state from {}".format(args.opt_state))
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    return optimizer
