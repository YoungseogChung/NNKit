import argparse, torch, sys

sys.path.append('../../')
from datasets.synth import *
from models.model import *

def parse_args():
    parser = argparse.ArgumentParser()

    """ general args """
    parser.add_argument('--parent_ep', type=int, default=1000,
                        help="max number of epochs to train for parent model")
    parser.add_argument('--ens_ep', type=int, default=300,
                        help="max number of epochs to train for ensemble")
    parser.add_argument('--cali', type=int, default=1,
                        help="use calibration loss")
    parser.add_argument('--sharp', type=int, default=1,
                        help="use sharpness loss")


    """ model args """
    parser.add_argument('--model_type', type=str, default='vanilla',
                        help="type of neural network to use")
    parser.add_argument('--hidden', type=int, default=10,
                        help="hidden dimension of neural networks")
    parser.add_argument('--bias', type=int, default=1,
                        help="1 to use bias in network weights")
    parser.add_argument('--num_layers', type=int, default=5,
                        help="number of layers in model")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate.")

    """ ensemble args """
    parser.add_argument('--num_ens', type=int, default=5,
                        help="number of models in ensemble")
    parser.add_argument('--dataset', type=str, default='para1',
                        help="dataset to use")
    parser.add_argument('--ood', type=float, default=0.3,
                        help="ood loss term coefficient")

    """ common data args """
    parser.add_argument('--noise', type=int, default=0,
                        help="add noise or not")

    """ train dataset """
    parser.add_argument('--batch', type=int, default=16,
                        help="training batch size")

    parser.add_argument('--train_size', type=int, default=100,
                        help="train dataset size")
    parser.add_argument('--train_mu', type=float, default=0.,
                        help="mean of oneD train data x")
    parser.add_argument('--train_sigma', type=float, default=3,
                        help="standard deviation of oneD train data x")
    parser.add_argument('--train_distr', type=str, default='eq',
                        help="train dataset distribution")
    parser.add_argument('--train_min', type=float, default=-5,
                        help="train dataset minimum x")
    parser.add_argument('--train_max', type=float, default=5,
                        help="train dataset maximum x")

    """ test dataset """
    parser.add_argument('--test_size', type=int, default=30,
                        help="test dataset size")
    parser.add_argument('--test_mu', type=float, default=0.,
                        help="mean of oneD test data x")
    parser.add_argument('--test_sigma', type=float, default=3,
                        help="standard deviation of oneD test data x")
    parser.add_argument('--test_distr', type=str, default='eq',
                        help="test dataset distribution")
    parser.add_argument('--test_min', type=float, default=-5,
                        help="test dataset minimum x")
    parser.add_argument('--test_max', type=float, default=5,
                        help="test dataset maximum x")


    """ misc args """
    parser.add_argument('--gpu', type=int, default=0,
                        help="set which cuda device to use")
    parser.add_argument('--seed', type=int, default=1234,
                        help="Random seed")

    # parser.add_argument('--', type=, default=,
    #                     help="")

    args = parser.parse_args()

    """ set model type """
    if args.model_type == 'vanilla':
        args.model = vanilla_nn
    elif args.model_type == 'pnn':
        args.model = prob_nn

    """ Set dataset method """
    if args.dataset == 'sine1':
        args.dataset_method = oneD_sine
    elif args.dataset == 'lin1':
        args.dataset_method = oneD_linear
    elif args.dataset == 'para1':
        args.dataset_method = oneD_parabola
    elif args.dataset == 'cubed1':
        args.dataset_method = oneD_cubed
    elif args.dataset == 'test1':
        args.dataset_method = oneD_test_1

    """ set data params"""
    args.train_data_params = {'batch_size': args.batch, 'shuffle': True,
                              'num_workers': 2}
    args.test_data_params = {'batch_size': args.test_size, 'shuffle': False,
                             'num_workers': 2}

    """ Set device """
    cuda_device = "cuda:{}".format(args.gpu)
    use_cuda = torch.cuda.is_available()
    device = torch.device(cuda_device if use_cuda else "cpu")
    print('Using device ', device)

    return args, device
