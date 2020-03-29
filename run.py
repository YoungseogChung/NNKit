import os, sys, random
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from utils.arg_parser import parse_exp_args
from utils.common import get_gpu_name
from utils.common import Logger

def create_dataloader(args, set_type):
    if set_type == 'train':
        print('Making training dataset loader')
        train_set = args.dataset_method(args.train_X_path, args.train_y_path)
        data_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True,
                                num_workers = args.num_cpu - 2, 
                                pin_memory = args.pin_memory)
    elif set_type == 'test':
        print('Making testing dataset loader')
        test_set = args.dataset_method(args.test_X_path, args.test_y_path)
        data_loader = DataLoader(dataset=test_set, batch_size=args.batch, shuffle=True,
                                num_workers = args.num_cpu - 2, 
                                pin_memory = args.pin_memory)
    return data_loader
   
def create_single_model(args):
    print('Creating model with\n \
           Input size: {}\n \
           Output size: {}\n \
           Activation {}\n \
           Num layers: {}\n \
           Hidden units per layer: {}\n \
           Using bias: {}\n \
           Using batchnorm {}\n \
           With batchsize {}'.format( \
           args.input_size, args.output_size, args.actv,
           args.num_layers, args.hidden, args.bias, args.bn, args.batch))
    model = args.model(input_size=args.input_size, output_size=args.output_size,
                       actv_type=args.actv,
                       num_layers=args.num_layers,
                       hidden_size=args.hidden, 
                       bias=args.bias, 
                       use_bn=args.bn)

    args.loss = model.loss
    if args.multi_gpu:
        print('Using data parallelism with {} GPUs'.format(args.num_gpu))
        model = nn.DataParallel(model, device_ids = args.device_ids)
    print('Sending model to device {}'.format(args.device))
    model.to(args.device)
    return model

def load_model(args):
    model_used_path = '/'.join((args.model_path).split('/')[:-1])
    model_used_args_path = os.path.join(model_used_path, 'args.txt') 
    model_used_args = parse_exp_args(model_used_args_path)
    print('Loading model from {}'.format(args.model_path))
    loaded_model = model_used_args.model(input_size=model_used_args.input_size, 
                                         output_size=model_used_args.output_size,
                                         actv_type=model_used_args.actv,
                                         num_layers=model_used_args.num_layers,
                                         hidden_size=model_used_args.hidden, 
                                         bias=model_used_args.bias, 
                                         use_bn=model_used_args.bn)

    # modify keys of loaded state dic
    loaded_state_dic = torch.load(args.model_path)

    if 'module' in list(loaded_state_dic.keys())[0]:
        mod_state_dic = OrderedDict()
        for k,v in loaded_state_dic.items(): 
            mod_k = k.strip('module.')
            if mod_k not in loaded_model.state_dict():
                print('skipping {}'.format(mod_k))
                continue
            mod_state_dic[mod_k] = loaded_state_dic[k]
        
    loaded_model.load_state_dict(mod_state_dic)
    del loaded_state_dic

    args.loss = loaded_model.loss

    if args.multi_gpu:
        print('Using data parallelism with {} GPUs'.format(args.num_gpu))
        loaded_model = nn.DataParallel(loaded_model)
    print('Sending model to device {}'.format(args.device))
    loaded_model.to(args.device)

    return loaded_model

def train_epoch(model, dataloader, optimizer, args):
    model.train()
    epoch_loss = 0
    for idx, (data, target) in tqdm(enumerate(dataloader)):
        data, target = data.to(args.device, non_blocking = args.non_blocking), \
                       target.to(args.device, non_blocking = args.non_blocking)

        output = model(data)

        loss = args.loss(output, target)
        optimizer.zero_grad()

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    return epoch_loss/idx

def test_epoch(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        len_pred = len(dataloader) * (dataloader.batch_size)
        out_pred = torch.FloatTensor(len_pred, args.output_size).fill_(0).to(args.device)
        loss_val = 0

        for idx, (data,target) in enumerate(dataloader):
            data, target = data.to(args.device, non_blocking=args.non_blocking), \
                           target.to(args.device, non_blocking=args.non_blocking)
            output = model(data)
            loss = args.loss(output, target)
            loss_val += loss.item()
            out_pred[output.size(0)*idx:output.size(0)*(1+idx)] = output.data
        loss_mean = loss_val/idx
        return loss_mean
        
def train(args, logger):
    """ make dataloader """
    train_loader = create_dataloader(args, 'train')
    test_loader = create_dataloader(args, 'test')

    """ make model and optimizer """
    if args.load_model:
        model = load_model(args)
    else:
        model = create_single_model(args)
    # TODO: make the optimizer an argument
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # TODO: make learning rate scheduler 

    print('Starting training...')
    for ep in tqdm(range(args.parent_ep)):
        ep_loss = train_epoch(model, train_loader, optimizer, args)
        logger.save_loss(type_of_loss='train', loss_val=ep_loss, epoch_num=ep)
        
        if ep % args.test_every == 0:
            print('Testing at epoch {}'.format(ep))
            test_loss = test_epoch(model, test_loader, args)
            logger.save_loss(type_of_loss='test', loss_val=test_loss, epoch_num=ep)

        if ep % args.save_model_every == 0:
            logger.save_model(model, ep)


def main():
    """ get run args """
    parser = argparse.ArgumentParser()
    #parser.add_argument('--o_dir', type=str, default='options',
    #                    help='dir containing options files')
    parser.add_argument('--o', type=str,
                        help='options file to use')
    parser.add_argument('--id', type=str,
                        help='ID of current experiment')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='ID of current experiment')
    parser.add_argument('--use_gpu', type=str, default=None,
                        help='list gpus you do not want to use')
    run_args = parser.parse_args()
        
    """ get exp args and  merge args"""
    args_file = run_args.o
    args = parse_exp_args(args_file)

    for k,v in vars(run_args).items():
        vars(args)[k] = v
    
    """ set seeds """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ multi GPU """
    CPU_COUNT = multiprocessing.cpu_count()
    GPU_COUNT = torch.cuda.device_count()
    print('Using device {}'.format(args.device))
    if 'cuda' in args.device.type:
        if args.use_gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu 
            gpu_list = args.use_gpu.split(',')
            args.device_ids = ['cuda:'+str(x) for x in gpu_list]
            GPU_COUNT = len(gpu_list)
        else:
            args.device_ids = ['cuda:'+str(x) for x in range(GPU_COUNT)]

        torch.backends.cudnn.benchmark = True
        if args.multi_gpu:
            _DEVICE = torch.device(args.device)
            args.batch *= GPU_COUNT
            print('Total batch size per iteration is now {}'.format(args.batch))

    args.num_cpu = CPU_COUNT
    args.num_gpu = GPU_COUNT


    """ make logger """
    print('Creating logger for log dir {}/{}'.format(args.log_dir, args.id))
    logger = Logger(args.id, args_file, args.log_dir, args)

    """ train """
    import pdb; pdb.set_trace()
    train(args, logger)

if __name__=='__main__':
    main()
