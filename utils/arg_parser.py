import os, sys, torch
import configparser as cfp
from argparse import Namespace
import numpy as np

sys.path.append('../')
from models.model import vanilla_nn, prob_nn
from utils.common import PlainDataset


def parse_exp_args(args_file):
    cf = cfp.ConfigParser()

    cf["DEFAULT"] = \
        {
          ### general args
          'parent_ep'   : 1000, # max number of epochs to train for parent model
          'ens_ep'      : 300,  # max number of epochs to train for ensemble
          'num_ens'     : 5,    # number of models in ensemble
          ### model args 
          'load_model'  : 0,    # 1 to load a model 
          'model_path'  : '',   # file path of model to load
          'model_type'  : 'vanilla', # type of neural network to use
          'actv'        : 'relu',    # activation to use for each layer
          'num_layers'  : 5,    # number of layers in model
          'hidden'      : 10,   # hidden dimension of neural networks
          'bias'        : 1,    # 1 to use bias in network weights
          'bn'          : 1,    # 1 to use batch normalization
          'lr'          : 1e-3, # learning rate
          ### dataset args
          'dataset_method' : 'plain',  # dataset class to use
          'dataset'        : 'data_1', # dataset data to use
          'batch'          : 16,       # training batch size
          'train_size'     : 100,      # train dataset size
          'test_size'      : 30,       # test dataset size
          ### gpu args
          'gpu'          : 0,   # set which cuda device to use
          'multi_gpu'    : 1,   # 1 to use multi_gpus
          'pin_memory'   : 1,   # 1 to pin memory 
          'non_blocking' : 1,   # 1 for non_blocking
          ### misc args
          'seed'             : 1234,  # Random seed
          'save_model_every' : 100,   # save the model every epochs
          'test_every'       : 100,   # test the model every epochs
        }

    cf.read(args_file)

    args = Namespace()
    """ general args """
    args.parent_ep = int(cf.get('args', 'parent_ep'))   
    args.ens_ep = int(cf.get('args', 'ens_ep'))   

    """ model args """
    args.load_model = bool(int(cf.get('args', 'load_model')))
    args.model_path = str(cf.get('args', 'model_path'))
    args.model_type = str(cf.get('args', 'model_type'))   
    args.actv = str(cf.get('args', 'actv'))
    args.num_layers = int(cf.get('args', 'num_layers'))   
    args.hidden = int(cf.get('args', 'hidden'))   
    args.bias = bool(int(cf.get('args', 'bias')))
    args.bn = bool(int(cf.get('args', 'bn')))
    args.lr = float(cf.get('args', 'lr'))   

    args.input_size = int(cf.get('args', 'input_size'))
    args.output_size = int(cf.get('args', 'output_size'))

    """ ensemble args """
    args.num_ens = int(cf.get('args', 'num_ens'))   
    args.dataset_method = cf.get('args', 'dataset_method')   
    args.dataset = str(cf.get('args', 'dataset')) 

    """ train dataset """
    args.batch = int(cf.get('args', 'batch'))   
    args.train_size = int(cf.get('args', 'train_size'))   

    """ test dataset """
    args.test_size = int(cf.get('args', 'test_size'))   

    """ misc args """
    args.gpu = int(cf.get('args', 'gpu'))   
    args.multi_gpu = bool(int(cf.get('args', 'multi_gpu'))) 
    args.pin_memory = bool(int(cf.get('args', 'pin_memory')))
    args.non_blocking = bool(int(cf.get('args', 'non_blocking')))
    args.seed = int(cf.get('args', 'seed'))   
    args.save_model_every = int(cf.get('args', 'save_model_every')) 
    args.test_every = int(cf.get('args', 'test_every')) 

    """ set model type """
    if args.model_type == 'vanilla':
        args.model = vanilla_nn
    elif args.model_type == 'pnn':
        args.model = prob_nn

    """ set datasets """
    if args.dataset_method == 'plain':
        args.dataset_method = PlainDataset

    args.train_X_path = os.path.join('data', args.dataset, 'train_X.npy')
    args.train_y_path = os.path.join('data', args.dataset, 'train_y.npy')
    args.test_X_path = os.path.join('data', args.dataset, 'test_X.npy')
    args.test_y_path = os.path.join('data', args.dataset, 'test_y.npy')
        
    """ set data params"""
    args.train_data_params = {'batch_size': args.batch, 'shuffle': True}
    args.test_data_params = {'batch_size': args.batch, 'shuffle': False}

    """ Set device """
    cuda_device = "cuda:{}".format(args.gpu)
    use_cuda = torch.cuda.is_available()
    device = torch.device(cuda_device if use_cuda else "cpu")
    args.device = device

    return args
