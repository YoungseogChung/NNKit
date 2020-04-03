from run import parse_run_args, parse_exp_args, parse_gpu_options, set_seeds
from run import print_model_specs, train

def train_ensemble(args, logger):
    """ make dataloader """
    if args.make_train_test:
        train_loader, test_loader = create_dataloader(args, None)
    else:
        train_loader = create_dataloader(args, 'train')
        test_loader = create_dataloader(args, 'test')

    """ make model and optimizer """
    if args.load_model:
        ens = [load_model(args) for _ in range(args.num_ens)]
    else:
        ens = [create_single_model(args) for _ in range(args.num_ens)]
    

    

