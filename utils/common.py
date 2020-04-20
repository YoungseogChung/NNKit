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

    def save_preds(self, pred_tensor, epoch_num):
        save_file = os.path.join(self.save_path, 'test_preds_ep_{}.npy'.format(epoch_num))
        np.save(save_file, pred_tensor.detach().cpu().numpy())


class PlainDataset(Dataset):
    def __init__(self, X_arr_path, y_arr_path, 
                 normalize, filter_outlier=False, args=None,
                 X_norm_stats=None, y_norm_stats=None, 
                 X_arr=None, y_arr=None):

        # 1) load the data
        if X_arr_path is not None and X_arr is None:
            X_arr = np.load(X_arr_path)
            y_arr = np.load(y_arr_path)

        # 2) filter outliers
        if filter_outlier:
            good_idx = self.filter_outlier(y_arr, args)
            X_arr = X_arr[good_idx]
            y_arr = y_arr[good_idx]
        
        # 3) normalize
        if normalize:
            print('Normalizing train features')
            self.X_mean, self.X_std = np.mean(X_arr, axis=0), np.std(X_arr, axis=0)+1e-10
            self.y_mean, self.y_std = np.mean(y_arr, axis=0), np.std(y_arr, axis=0)
            X_arr = (X_arr - self.X_mean)/self.X_std
            y_arr = (y_arr - self.y_mean)/self.y_std
        if X_norm_stats is not None:
            print('Normalizing test features with training features stats')
            X_mean, X_std = X_norm_stats
            y_mean, y_std = y_norm_stats
            X_arr = (X_arr - X_mean)/X_std
            y_arr = (y_arr - y_mean)/y_std

        # 4) load into tensors
        print('Loading data into datasets')
        self.X = torch.FloatTensor(X_arr)
        num_items = self.X.shape[0]
        self.y = torch.FloatTensor(y_arr).reshape(num_items,-1)

        print('Loaded {} datapoints'.format(num_items))

    def filter_outlier(self, y_arr, args):
        import pudb; pudb.set_trace()
        percent = 0.01
        y_ceil = np.percentile(y_arr, 100-percent, axis=0)
        y_floor = np.percentile(y_arr, percent, axis=0)


        good_idx = ((y_arr>y_floor).flatten())*((y_arr<y_ceil).flatten())
        print('Filtering {} points from total {} points, for {} points'.format(
                y_arr.size-np.sum(good_idx), y_arr.size, np.sum(good_idx)))

        return good_idx

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return X,y 

