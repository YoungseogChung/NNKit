import sys
import os
import pickle
import random
import xarray
import numpy as np
from tqdm import tqdm

data_path = './'
train_fn = 'full_train.pkl'
test_fn = 'full_test.pkl'
TEST_FRAC = 0.2

def load_data(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def process_shot(shot_data):
    # TODO: I guess we should check the data are regularly spaced but I'll do it later
    shot_data = shot_data.drop('shot').squeeze()
    good_times = []
    times = shot_data['time']
    prev_good = False
    prev_time = None
    for i, time in enumerate(times):
        data = shot_data.sel(time=time)
        # TODO: add any other data quality checks here
        good = np.all(np.isfinite(data).to_array().values)
        if good and prev_good:
            good_times.append(prev_time)
        prev_good = good
        prev_time = i
    data = shot_data.to_array().values
    var_names = shot_data.variables.keys()[1:]
    rows = dict(zip(var_names, range(len(var_names))))
    return data, good_times, rows

def add_disruption_data(shot_data, is_disrupting):
    if is_disrupting:
        shot_data["disrupted"] = shot_data["time"] >= 0
    else:
        # very hacky way to get all False, I don't wanna learn this library
        shot_data["disrupted"] = shot_data["time"] == "boob"
    return shot_data


def split_shots(data):
    shots = data.keys()
    random.shuffle(shots)
    test_shot_cutoff = int(len(shots) * TEST_FRAC)
    test_shots = shots[:test_shot_cutoff]
    train_shots = shots[test_shot_cutoff:]
    return train_shots, test_shots

def main(dis_fn, nondis_fn):
    dis_data = load_data(dis_fn)
    nondis_data = load_data(nondis_fn)
    dis_train_shots, dis_test_shots = split_shots(dis_data)
    nondis_train_shots, nondis_test_shots = split_shots(nondis_data)
    train_dataset = {}
    print("Unpacking Disrupting Train Set")
    for shot in tqdm(dis_train_shots):
        shot_data = dis_data[shot]
        shot_data = add_disruption_data(shot_data, True)
        shot_data, time_list, name_list = process_shot(shot_data)
        train_dataset[shot] = (time_list, shot_data)
    print("Unpacking Nondisrupting Train Set")
    for shot in tqdm(nondis_train_shots):
        shot_data = nondis_data[shot]
        shot_data = add_disruption_data(shot_data, False)
        shot_data, time_list, name_list = process_shot(shot_data)
        train_dataset[shot] = (time_list, shot_data)
    train_dataset = (name_list, train_dataset)
    test_dataset = {}
    print("Unpacking Disrupting Test Set")
    for shot in tqdm(dis_test_shots):
        shot_data = dis_data[shot]
        shot_data = add_disruption_data(shot_data, True)
        shot_data, time_list, name_list = process_shot(shot_data)
        train_dataset[shot] = (time_list, shot_data)
    print("Unpacking Nondisrupting Test Set")
    for shot in tqdm(nondis_test_shots):
        shot_data = nondis_data[shot]
        shot_data = add_disruption_data(shot_data, False)
        shot_data, time_list, name_list = process_shot(shot_data)
        train_dataset[shot] = (time_list, shot_data)
    test_dataset = (name_list, test_dataset)
    train_path = os.path.join(data_path, train_fn)
    test_path = os.path.join(data_path, test_fn)
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
