import os, sys, torch
import configparser as cfp
from argparse import Namespace
import numpy as np

sys.path.append('../')
from models.model import vanilla_nn, prob_nn, cnn
from utils.common import PlainDataset


def parse_exp_args(args_file, use_gpu):
    cf = cfp.ConfigParser()

    cf["DEFAULT"] = \
        {
          ### general args
          'parent_ep'   : 1000, # max number of epochs to train for parent model
          'ens_ep'      : 300,  # max number of epochs to train for ensemble
          'num_ens'     : 5,    # number of models in ensemble
          ### model args
          'arch'  : '',   # declaring model architechture
          'load_model'  : 0,    # 1 to load a model
          'model_path'  : '',   # file path of model to load
          'model_type'  : 'vanilla', # type of neural network to use
          'actv'        : 'relu',    # activation to use for each layer
          'num_layers'  : 5,    # number of layers in model
          'hidden'      : 10,   # hidden dimension of neural networks
          'bias'        : 1,    # 1 to use bias in network weights
          'bn'          : 1,    # 1 to use batch normalization
          ### optimizer args
          'loss_factor' : 1,       # constant to multiply to loss
          'optimizer'   : 'adam',  # type of optimzier to use
          'lr'          : 1e-3,    # learning rate
          'lr_scheduler': 1,       # 1 to use adaptive lr scheduler
          'patience'    : 1,       # patience parameter for lr scheduler
          'momentum'    : 0.9,     # momentum parameter for optimizer (SGD)
          ### dataset args
          'dataset_method' : 'plain',  # dataset class to use
          'make_train_test': 0,        # 1 to make train test split
          'dataset'        : 'data_1', # dataset data to use
          'normalize'      : 1,        # 1 to normalize training features
          'batch'          : 16,       # training batch size
          'train_size'     : 100,      # train dataset size
          'test_size'      : 30,       # test dataset size
          ### gpu args
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
    args.parent_ep = int(cf.get('general', 'parent_ep'))
    args.ens_ep = int(cf.get('general', 'ens_ep'))

    """ model args """
    args.model_type = str(cf.get('model', 'model_type'))
    args.model_arch = str(cf.get('model', 'arch'))
    args.load_model = bool(int(cf.get('model', 'load_model')))
    args.model_path = str(cf.get('model', 'model_path'))
    args.output_size = int(cf.get('model', 'output_size'))
    if len(args.model_arch) >= 1:
        args.in_channels = int(cf.get('model', 'in_channels'))
        args.in_length = int(cf.get('model', 'in_length'))
    if len(args.model_arch) < 1:
        args.input_size = int(cf.get('model', 'input_size'))
        args.actv = str(cf.get('model', 'actv'))
        args.num_layers = int(cf.get('model', 'num_layers'))
        args.hidden = int(cf.get('model', 'hidden'))
        args.bias = bool(int(cf.get('model', 'bias')))
        args.bn = bool(int(cf.get('model', 'bn')))

    """ optimizer args """
    args.loss_factor = float(cf.get('optimizer', 'loss_factor'))
    args.optimizer = str(cf.get('optimizer', 'opt_method'))
    args.lr = float(cf.get('optimizer', 'lr'))
    args.lr_scheduler = str(cf.get('optimizer', 'lr_scheduler'))
    args.patience = int(cf.get('optimizer', 'patience'))
    args.momentum = float(cf.get('optimizer', 'momentum'))

    """ ensemble args """
    if 'ens' in args.model_type:
        args.num_ens = int(cf.get('ensemble', 'num_ens'))

    """ dataset args """
    args.dataset_method = cf.get('dataset', 'dataset_method')
    args.dataset = str(cf.get('dataset', 'dataset'))
    args.normalize = bool(int(cf.get('dataset', 'normalize')))
    args.batch = int(cf.get('dataset', 'batch'))
    args.train_size = int(cf.get('dataset', 'train_size'))
    args.test_size = int(cf.get('dataset', 'test_size'))
    args.make_train_test = bool(int(cf.get('dataset','make_train_test')))
    if args.make_train_test:
        args.test_prop = float(cf.get('dataset','test_prop'))

    """ gpu args """
    # args.multi_gpu = bool(int(cf.get('gpu', 'multi_gpu')))
    args.expand_batch = bool(int(cf.get('gpu', 'expand_batch')))
    args.pin_memory = bool(int(cf.get('gpu', 'pin_memory')))
    args.non_blocking = bool(int(cf.get('gpu', 'non_blocking')))

    """ misc args """
    args.seed = int(cf.get('misc', 'seed'))
    args.save_model_every = int(cf.get('misc', 'save_model_every'))
    args.test_every = int(cf.get('misc', 'test_every'))

    ### done gathering args, now setting them ###

    """ set model type """
    if args.model_type == 'vanilla':
        args.model = vanilla_nn
    elif args.model_type == 'pnn':
        args.model = prob_nn
    elif args.model_type == 'cnn':
        args.model = cnn

    """ set datasets """
    if args.dataset_method == 'plain':
        args.dataset_method = PlainDataset

    if args.make_train_test:
        args.X_path = os.path.join('data', args.dataset,'X.npy')
        args.y_path = os.path.join('data', args.dataset, 'y.npy')
    else:
        args.train_X_path = os.path.join('data', args.dataset, 'train_X.npy')
        args.train_y_path = os.path.join('data', args.dataset, 'train_y.npy')
        args.test_X_path = os.path.join('data', args.dataset, 'test_X.npy')
        args.test_y_path = os.path.join('data', args.dataset, 'test_y.npy')

    """ set data params"""
    args.train_data_params = {'batch_size': args.batch, 'shuffle': True}
    args.test_data_params = {'batch_size': args.batch, 'shuffle': False}

    """ Set device """
    use_cuda = torch.cuda.is_available()
    if use_gpu is not None:
        # args.gpu_list = [int(x) for x in (use_gpu.split(','))]
        # cuda_device = 'cuda:{}'.format(args.gpu_list[0])
        num_use_gpu = len(use_gpu.split(','))
        # actual physical gpu devices
        args.gpu_device_list = [int(x) for x in (use_gpu.split(','))]
        # virtual name of gpu devices
        args.gpu_name_list = [int(x) for x in range(num_use_gpu)]

        if num_use_gpu == 1:
            args.multi_gpu = False
        else:
            args.multi_gpu = True

    cuda_device = 'cuda:0'
    device = torch.device(cuda_device if use_cuda else 'cpu')
    args.device = device

    return args
