import sys
import pickle as pkl
import random
import numpy as np
from copy import deepcopy


# TODO: make these constants into arguments
SIGNAL_LIST = ['R0', 'aminor', 'dssdenest', 'efsbetan', 'efsli', 
               'efsvolume', 'ip', 'kappa', 'tribot', 'tritop', 
               'pinj']

               #'pinj', 'pinj_15l', 'pinj_15r', 'pinj_21l', 'pinj_21r', 
               #'pinj_30l', 'pinj_30r', 'pinj_33l', 'pinj_33r']

               #'pinj', 'pinj_15l', 'pinj_15r', 'pinj_21l', 'pinj_21r', 
               #'pinj_30l', 'pinj_30r', 'pinj_33l', 'pinj_33r']

NUM_DIS = 500
NUM_NONDIS = 500
TEST_FRAC = 0.1
START_TIME = -2000
END_TIME = -10

def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    if 'DB' in data.keys():
        data = data['DB']
    return data

def split_shots(data, num_shots):
    curr_shots = deepcopy(data.keys())
    assert num_shots <= len(curr_shots)
    while True:
        if num_shots > len(curr_shots):
            raise RuntimeError('number of shots too many for number of good shots')
        shots = np.random.choice(curr_shots, size=num_shots, replace=False)
        inspect_shot_signals = []
        for shot in shots:
            curr_signals = list(data[shot].keys())
            has_all_signals = True
            for signal in SIGNAL_LIST:
                if signal not in curr_signals:
                    has_all_signals = False
                    print('shot {} does not have signal {}'.format(shot, signal))
                    curr_shots.remove(shot)
                    break
            inspect_shot_signals.append(has_all_signals)
        if np.array(inspect_shot_signals).all():
            print('list of shots is all good')
            break
        else:
            print('list of shots is not good, resampling')

    random.shuffle(shots)
    test_shot_cutoff = int(len(shots) * TEST_FRAC)
    test_shots = shots[:test_shot_cutoff]
    train_shots = shots[test_shot_cutoff:]
    return train_shots, test_shots

# train_dis_data = process_data(dis_signal_filtered, dis_train_shots)
def process_data(data, shots):
    processed_data = {}
    for shot in shots:
        shot_data = data[shot][SIGNAL_LIST].squeeze()
        time_np = shot_data['time'].data
        shot_data_np = shot_data.to_array().data.squeeze()
        # first find start time idx
        for time_idx in range(time_np.size-1):
            if time_np[time_idx] <= START_TIME and START_TIME < time_np[time_idx+1]:
                start_time_idx = time_idx
                break
        # second find end time idx
        for time_idx in range(time_np.size-1):
            if time_np[time_idx] < END_TIME and END_TIME <= time_np[time_idx+1]:
                end_time_idx = time_idx+1
                break

        # make sure all signals are in the subset of data
        assert shot_data_np.shape[0] == len(SIGNAL_LIST)

        # scan through time to get subset of data that is not ana
        for idx in range(start_time_idx, end_time_idx):
            subset_by_time = shot_data_np[:,idx:end_time_idx]
            if np.isfinite(subset_by_time).all():
                time_range = time_np[idx:end_time_idx]
                break
            assert subset_by_time.size > 0
        if idx == end_time_idx-1:
            print('Shot {} had no time subset without nans'.format(shot))
            continue
        processed_data[shot] = (time_range, subset_by_time)

    return processed_data
            

def main(dis_pkl, nondis_pkl):
    print('loading dis data')
    dis_data = load_data(dis_pkl)
    print('loading nondis data')
    nondis_data = load_data(nondis_pkl)
    #dis_signal_filtered = dis_data[SIGNAL_LIST]
    #nondis_signal_filtered = nondis_data[SIGNAL_LIST]

    dis_train_shots, dis_test_shots = split_shots(dis_data, NUM_DIS)
    nondis_train_shots, nondis_test_shots = split_shots(nondis_data, NUM_NONDIS)

    print('processing dis data')
    train_dis_data = process_data(dis_data, dis_train_shots)
    test_dis_data = process_data(dis_data, dis_test_shots)
    print('processing nondis data')
    train_nondis_data = process_data(nondis_data, nondis_train_shots)
    test_nondis_data = process_data(nondis_data, nondis_test_shots)

    print('done processing, saving data')
    with open('train_dis.pkl', 'wb') as f:
        pkl.dump(train_dis_data, f)
    with open('test_dis.pkl', 'wb') as f:
        pkl.dump(test_dis_data, f)
    with open('train_nondis.pkl', 'wb') as f:
        pkl.dump(train_nondis_data, f)
    with open('test_nondis.pkl', 'wb') as f:
        pkl.dump(test_nondis_data, f)

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])

