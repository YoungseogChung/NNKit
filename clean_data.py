import os, sys
import numpy as np

def main(dataset, new_dataset, thresh):
    import pdb; pdb.set_trace()
    data_path = dataset
    assert(os.path.exists(data_path))

    new_data_path = new_dataset
    os.mkdir(new_data_path)

    train_X= np.load(os.path.join(data_path, 'train_X.npy'))
    train_y= np.load(os.path.join(data_path, 'train_y.npy'))
    test_X= np.load(os.path.join(data_path, 'test_X.npy'))
    test_y= np.load(os.path.join(data_path, 'test_y.npy'))

    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    #train_X = np.load('train_X_path')
    #train_y = np.load('train_y_path')
    #test_X = np.load('test_X_path')
    #test_y = np.load('test_y_path')

    train_data = ('train', train_X, train_y)
    test_data = ('test', test_X, test_y)

    for data in [train_data, test_data]:
        label,X,y = data
        bad_idxs = []
        for idx in range(y.shape[0]):
            if np.abs(y[idx]) > thresh:
                bad_idxs.append(idx)
        good_idxs = [i for i in range(y.shape[0]) if i not in bad_idxs]
        good_X = X[good_idxs]
        good_y = y[good_idxs]
        np.save(os.path.join(new_dataset, '{}_X.npy'.format(label)), good_X)
        np.save(os.path.join(new_dataset, '{}_y.npy'.format(label)), good_y)

if __name__=='__main__':
    ### arguments
    ### 1. datadir to modify
    ### 2. new dataset name
    ### 3. threshold for values
    main(str(sys.argv[1]), str(sys.argv[2]), float(sys.argv[3]))

