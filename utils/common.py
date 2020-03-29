import os, subprocess
import numpy as np
from shutil import copyfile

import torch
from torch.utils.data import Dataset

def get_gpu_name():
    try:
        out_str = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode('utf-8').split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)
        
class Logger():
    def __init__(self, logger_id, args_file, log_dir, args):
        self.id = logger_id
        self.save_path = os.path.join(log_dir, self.id)
        self.args_file = args_file
        if os.path.exists(self.save_path):
            raise RuntimeError('ID {} already exists'.format(self.save_path))
        os.mkdir(self.save_path)

        # if model was loaded, record wehre the loaded model came from
        if args.load_model:
            with open(os.path.join(self.save_path, 'model_loaded.txt'), 'w') as f:
                f.write('Training began with model loaded from {}'.format(args.model_path))

        # copy the args file to save directory
        args_file_name = (self.args_file.split('/'))[-1]
        copyfile(self.args_file, os.path.join(self.save_path, 'args.txt')) 
        #with open(os.path.join(self.save_path, 'args.txt'), 'w') as f:
        #    f.write(self.args_file)
        
        self.train_loss_file = os.path.join(self.save_path, 'train_loss.txt')
        self.val_loss_file = os.path.join(self.save_path, 'val_loss.txt')
        self.test_loss_file = os.path.join(self.save_path, 'test_loss.txt')
        
    def save_loss(self, type_of_loss, loss_val, epoch_num):
        if type_of_loss == 'train':
            loss_file = self.train_loss_file
        elif type_of_loss == 'val':
            loss_file = self.val_loss_file
        elif type_of_loss == 'test':
            loss_file = self.test_loss_file
        
        with open(loss_file, 'a+') as f:
            f.seek(0)
            data = f.read(100)
            if len(data) > 0:
                f.write('\n')
            f.write('{}, {}'.format(epoch_num, loss_val))
        
    def save_model(self, model, epoch_num):
        save_file = os.path.join(self.save_path, 'model_ep_{}'.format(epoch_num))
        model.eval()
        torch.save(model.state_dict(), save_file)
    

class PlainDataset(Dataset):
    def __init__(self, X_arr_path, y_arr_path):
        X_arr = np.load(X_arr_path)
        y_arr = np.load(y_arr_path)
        print('Loading data into datasets')
        self.X = torch.FloatTensor(X_arr)
        self.y = torch.FloatTensor(y_arr).reshape(-1,1)
        print('Loaded {} datapoints'.format(self.y.shape[0]))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X,y 

