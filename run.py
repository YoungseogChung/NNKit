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
from torch.nn.parallel import gather

from utils.arg_parser import parse_exp_args
from utils.common import get_gpu_name
from utils.common import Logger

def create_dataloader(args, set_type):
    if args.make_train_test:
        X_arr = np.load(args.X_path)
        y_arr = np.load(args.y_path)
        num_items = y_arr.shape[0]
        num_test = int(num_items*args.test_prop)
        num_train = num_items - num_test
        test_idxs = np.random.choice(num_items, size=num_test, replace=False)
        train_idxs = np.array([x for x in range(num_items) if x not in test_idxs])
    
        train_X = X_arr[train_idxs]
        train_y = y_arr[train_idxs]
        test_X = X_arr[test_idxs]
        test_y = y_arr[test_idxs]
        
        print('Making training dataset loader')
        train_set = args.dataset_method(None, None, train_X, train_y)
        train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True,
                                       num_workers = args.num_cpu - 2, 
                                      pin_memory = args.pin_memory)
        print('Making testing dataset loader')
        test_set = args.dataset_method(None, None, test_X, test_y)
        test_data_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False,
                                      num_workers = args.num_cpu - 2, 
                                      pin_memory = args.pin_memory)

        return train_data_loader, test_data_loader

    if set_type == 'train':
        print('Making training dataset loader')
        train_set = args.dataset_method(args.train_X_path, args.train_y_path, 
                                        normalize=args.normalize)
        if args.normalize:
            args.X_mean, args.X_std = train_set.X_mean, train_set.X_std
            args.y_mean, args.y_std = train_set.y_mean, train_set.y_std 
        data_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True,
                                num_workers = args.num_cpu - 2, 
                                pin_memory = args.pin_memory)
    elif set_type == 'test':
        print('Making testing dataset loader')
        if args.normalize:
            X_norm_stats = (args.X_mean, args.X_std)
            y_norm_stats = (args.y_mean, args.y_std)
        else:
            X_norm_stats, y_norm_stats = None, None

        test_set = args.dataset_method(args.test_X_path, args.test_y_path, 
                                       normalize = False,
                                       X_norm_stats = X_norm_stats,
                                       y_norm_stats = y_norm_stats)
        data_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False,
                                num_workers = args.num_cpu - 2, 
                                pin_memory = args.pin_memory)
        
    return data_loader
   
def create_single_model(args):
    print('Creating model')
    if len(args.model_arch) < 1:
        model = args.model(input_size=args.input_size, output_size=args.output_size,
                           actv_type=args.actv,
                           num_layers=args.num_layers,
                           hidden_size=args.hidden, 
                           bias=args.bias, 
                           use_bn=args.bn)
    else:
        model = args.model(arch_str=args.model_arch, in_channels=args.in_channels,
                           in_length = args.in_length)

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
    if model_used_args.model_type in ['vanilla', 'pnn']:
        loaded_model = model_used_args.model(input_size=model_used_args.input_size, 
                                             output_size=model_used_args.output_size,
                                             actv_type=model_used_args.actv,
                                             num_layers=model_used_args.num_layers,
                                             hidden_size=model_used_args.hidden, 
                                             bias=model_used_args.bias, 
                                             use_bn=model_used_args.bn)
    elif model_used_args.model_type in ['cnn']:
        loaded_model = model_used_args.model(arch_str=model_used_args.model_arch, 
                                             in_channels=model_used_args.in_channels, 
                                             in_length=model_used_args.in_length)

    # modify keys of loaded state dic
    loaded_state_dic = torch.load(args.model_path)

    if 'module' in list(loaded_state_dic.keys())[0]:
        mod_state_dic = OrderedDict()
        for k,v in loaded_state_dic.items(): 
            mod_k = k.replace('module.', '')
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
    num_items = 0
    for idx, (data, target) in tqdm(enumerate(dataloader)):
        data, target = data.to(args.device, non_blocking = args.non_blocking), \
                       target.to(args.device, non_blocking = args.non_blocking)

        output = model(data)

        loss = args.loss_factor*args.loss(output, target)
        optimizer.zero_grad()
        epoch_loss += (loss.item())*data.shape[0]
        num_items += data.shape[0]
        loss.backward()
        optimizer.step()

    return epoch_loss/(num_items)

def test_epoch(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        len_pred = len(dataloader) * (dataloader.batch_size)
        out_pred = torch.FloatTensor(len_pred, args.output_size).fill_(0).to(args.device)
        loss_val = 0
        num_items = 0

        for idx, (data,target) in enumerate(dataloader):
            data, target = data.to(args.device, non_blocking=args.non_blocking), \
                           target.to(args.device, non_blocking=args.non_blocking)
            output = model(data)
            loss = args.loss(output, target)
            loss_val += loss.item()*data.shape[0]
            num_items += data.shape[0]
            out_pred[output.size(0)*idx:output.size(0)*(1+idx)] = output.data
        loss_mean = loss_val/(num_items)
        return loss_mean, out_pred
        
def train(args, model, logger):

    """ make dataloader """
    if args.make_train_test:
        train_loader, test_loader = create_dataloader(args, None)
    else:
        train_loader = create_dataloader(args, 'train')
        test_loader = create_dataloader(args, 'test')

    """ make optimizer """
    # TODO: make the optimizer an argument
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # TODO: make learning rate scheduler 
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, 
                                                        verbose=True)


    print('Starting training...')
    for ep in tqdm(range(args.parent_ep)):
        train_loss = train_epoch(model, train_loader, optimizer, args)
        logger.save_loss(type_of_loss='train', loss_val=train_loss, epoch_num=ep)
        lr_scheduler.step(train_loss)
        
        if ep % args.test_every == 0:
            print('Testing at epoch {}'.format(ep))
            test_loss, test_preds = test_epoch(model, test_loader, args)
            logger.save_loss(type_of_loss='test', loss_val=test_loss, epoch_num=ep)
            logger.save_preds(pred_tensor=test_preds, epoch_num=ep)
            #lr_scheduler.step(test_loss)
            print('EP {0} train_loss {1:.3f}, test loss {2:.3f}'.format(ep, train_loss, test_loss))

        if ep % args.save_model_every == 0:
            logger.save_model(model, ep)

        if optimizer.param_groups[0]['lr'] < 1e-7:
            print('lr minimum reached, terminating training')
            break

def parse_run_args():
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

    return run_args

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_gpu_options(args):
    CPU_COUNT = multiprocessing.cpu_count()
    GPU_COUNT = torch.cuda.device_count()
    print('Using device {}'.format(args.device))
    if 'cuda' in args.device.type:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        if args.use_gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu 
            gpu_list = args.use_gpu.split(',')
            args.device_ids = ['cuda:'+str(x) for x in gpu_list]
            GPU_COUNT = len(gpu_list)
        else:
            args.device_ids = ['cuda:'+str(x) for x in range(GPU_COUNT)]

        if args.multi_gpu:
            #_DEVICE = torch.device(args.device)
            args.device = args.device_ids[0]
            if args.expand_batch:
                args.batch *= GPU_COUNT
            print('Total batch size per iteration is now {}'.format(args.batch))

    args.num_cpu = CPU_COUNT
    args.num_gpu = GPU_COUNT

    return args

def print_model_specs(args):
    print(('Launching training with {} model with\n' 
           '  Input size: {}\n'
           '  Output size: {}\n'
           '  Activation {}\n'
           '  Num hidden layers: {}\n'
           '  Hidden units per layer: {}\n'
           '  Using bias: {}\n'
           '  Using batchnorm {}\n'
           '  With batchsize {}').format( \
           args.model_type, args.input_size, args.output_size, args.actv,
           args.num_layers, args.hidden, args.bias, args.bn, args.batch))



def main():
    """ get run args """
    run_args = parse_run_args()
        
    """ get exp args and  merge args"""
    args_file = run_args.o
    args = parse_exp_args(args_file)

    for k,v in vars(run_args).items():
        vars(args)[k] = v
    #print(args)
    
    """ set seeds """
    set_seeds(args.seed)

    """ multi GPU """
    args = parse_gpu_options(args)

    """ print model type """
    if args.model_type in ['vanilla', 'pnn']:
        print_model_specs(args)

    """ check args """
    print(args)
    import pdb; pdb.set_trace()

    """ make model """
    if args.load_model:
        model = load_model(args)
    else:
        model = create_single_model(args)

    """ check before launch """
    import pdb; pdb.set_trace()

    """ make logger """
    print('Creating logger for log dir {}/{}'.format(args.log_dir, args.id))
    logger = Logger(args.id, args_file, args.log_dir, args)

    """ launch training """
    train(args, model, logger)

if __name__=='__main__':
    main()
