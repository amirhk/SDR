# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 14:24:38 2017

@author: sbanijam
"""
import plane_obstacles_mdp
import pickle
import os
import numpy as np
import time


def get_plane_dataset(filename,
                      num_samples=10000,
                      W=40, H=40,
                      overwrite_datasets=True,
                      **kwargs):
    """
    Create a dataset for the Plane with obstacles MDP.
    """
    if not os.path.exists(filename) or overwrite_datasets:
        print('Creating plane dataset', filename)
        mdp = plane_obstacles_mdp.PlaneObstaclesMDP(H=H, W=W)

        X = np.zeros((num_samples, 1, W, H), dtype='float32')
        C = np.zeros((num_samples, 2), dtype='float32')

        for i in range(int(num_samples/2)):
            print(i)
            s = mdp.sample_random_state()
            X[i, :] = s[1]
            C[i,:] = np.asarray([0,1])

        for i in range(int(num_samples/2)):
            print(i + int(num_samples/2))
            s = mdp.sample_random_state()
            X[i + int(num_samples/2), :] = mdp.render(s[0],'cross')
            C[i + int(num_samples/2), :] = np.asarray([1,0])

        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        X = X[idx]
        C = C[idx]

        pickle.dump((X,C), open(filename, 'wb'))
        return X
    else:
        print('Loading plane dataset', filename)
        return pickle.load(open(filename, 'rb'))

get_plane_dataset('plane_dataset_train', num_samples = 10000)
get_plane_dataset('plane_dataset_test', num_samples = 100)

#get_pole_dataset('pole_dataset_train')
#get_cartpole_dataset('cartpole_dataset_train')

#get_arm_dataset('arm_dataset_train')

#get_pole_dataset_with_noise('pole_dataset_with_noise')
