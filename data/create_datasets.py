import sys, os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: make these into arguments
WINDOW_LEN = 100
#WINDOW_SAMPLING_NUM = 10
#WINDOW_SAMPLING_NUM = 20
WINDOW_SAMPLING_NUM = 5
PRED_INTERVAL = 250
ACT_SIGNAL = ['pinj']
TARGET_SIGNAL = ['efsbetan']
#TARGET_TYPE = 'raw'
#TARGET_TYPE = 'pdelta'
TARGET_TYPE = 'delta'
SIGNAL_LIST = ['R0', 'aminor', 'dssdenest', 'efsbetan', 'efsli', 
               'efsvolume', 'ip', 'kappa', 'tribot', 'tritop', 
               'pinj']

if WINDOW_LEN % WINDOW_SAMPLING_NUM != 0:
    print('Window sampling interval will not be exactly equispaced!')

def find_tidx(target_t, t_array):
    t_array = t_array.flatten()
    if target_t > t_array[-1]:
        raise RuntimeError('Target time exceeded array')
    
    if target_t == t_array[0]:
        return 0
    
    target_idx = None
    for t_idx in range(t_array.size - 1):
        if t_array[t_idx] < target_t and target_t <= t_array[t_idx+1]:
            target_idx = t_idx+1
            break
    if target_idx is None:
        print('Error in finding t_idx: target {} in time arr from {} to {}'.format(
        target_t, t_array[0], t_array[-1]))
    return target_idx

def make_prediction_data(dataset_list, act_idx, target_idx):
    sar_list = []
    X_list = []
    y_list = []
    for dataset in dataset_list:
        for shot,v in tqdm(dataset.items()):
            times = v[0].flatten()
            if times.size == 0:
                print(shot)
                continue
            values = v[1]
            window_beg_time = times[0]
            
            

            while True:
                window_end_time = window_beg_time + WINDOW_LEN
                pred_time = window_end_time + PRED_INTERVAL
                if pred_time > times[-1]:
                    break

                window_beg_idx = find_tidx(window_beg_time, times)
                window_end_idx = find_tidx(window_end_time, times)
                pred_time_idx = find_tidx(pred_time, times)

                try:
                    window = values[:, window_beg_idx:window_end_idx]
                    action = np.mean(values[act_idx, window_end_idx : pred_time_idx], axis = 1)
                    orig_target = values[target_idx, window_end_idx-1]
                    target = values[target_idx, pred_time_idx]
                    target_delta = target-orig_target
                    if orig_target == 0:
                        print(shot, orig_target, target_delta)
                    target_pdelta = target_delta/(orig_target + (np.sign(orig_target)*1e-10))

                except:
                    print(shot, values.shape, window_beg_idx, window_end_idx, pred_time_idx)
                    raise RuntimeError
                window_sample = window[:,np.linspace(0, WINDOW_LEN-1, WINDOW_SAMPLING_NUM).astype(int)]

                curr_sasr = (window_sample, action, target)
                curr_X = np.concatenate([window_sample.flatten(), action.flatten()]).reshape(1,-1)
                assert curr_X.shape[0] == 1
                if TARGET_TYPE=='raw':
                    curr_y = target.flatten().reshape(1,-1)
                elif TARGET_TYPE=='delta':
                    curr_y = target_delta.flatten().reshape(1,-1)
                elif TARGET_TYPE=='pdelta':
                    curr_y = target_pdelta.flatten().reshape(1,-1)
                else:
                    raise ValueError('TARGET_TYPE must be one of raw, delta or pdelta')


                sar_list.append(curr_sasr)
                X_list.append(curr_X)
                y_list.append(curr_y)

                window_beg_time += 10
    X_npy = np.concatenate(X_list, axis=0)
    y_npy = np.concatenate(y_list, axis=0)
    
    return X_npy, y_npy, sar_list


def main(data_dir):

    if os.path.exists(data_dir):
        raise RuntimeError('data directory already exists')
    os.mkdir(data_dir)
    train_dis = pkl.load(open('train_dis.pkl', 'rb'), encoding='latin1')
    test_dis = pkl.load(open('test_dis.pkl', 'rb'), encoding='latin1')
    train_nondis = pkl.load(open('train_nondis.pkl', 'rb'), encoding='latin1')
    test_nondis = pkl.load(open('test_nondis.pkl', 'rb'), encoding='latin1')
    train_datasets = [train_dis, train_nondis]
    test_datasets = [test_dis, test_nondis]
    
    act_idx = [SIGNAL_LIST.index(x) for x in ACT_SIGNAL]
    target_idx = [SIGNAL_LIST.index(x) for x in TARGET_SIGNAL]
    
    train_X, train_y, train_sar = make_prediction_data([train_dis, train_nondis], act_idx, target_idx)
    test_X, test_y, test_sar = make_prediction_data([test_dis, test_nondis], act_idx, target_idx)
    
    np.save(os.path.join(data_dir,'train_X.npy'), train_X)
    np.save(os.path.join(data_dir,'train_y.npy'), train_y)
    np.save(os.path.join(data_dir,'test_X.npy'), test_X)
    np.save(os.path.join(data_dir,'test_y.npy'), test_y)
    pkl.dump(train_sar, open(os.path.join(data_dir, 'train_sar.pkl'), 'wb'))
    pkl.dump(test_sar, open(os.path.join(data_dir, 'test_sar.pkl'), 'wb'))



if __name__=='__main__':
    main(sys.argv[1])
