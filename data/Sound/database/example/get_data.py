import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import pickle
import os


def write_data(data, path):
    # store list in binary file so 'wb' mode
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)
        print('Done writing list into a binary file')


def read_data(path):
    # for reading also binary mode is important
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data
    

def read_data_from_dir(folder_path):
    extension = '.bin' 
    files = [f for f in os.listdir(folder_path) 
         if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(extension)]
    data_path ='{}/{}'
    data = []
    for file in files:
        data += read_data(data_path.format(folder_path, file))
    return data

def read_data_full():
    folder_path = 'D:/Phd/sound_db/database{}'
    motors_path = '/motors/samples'
    airplane_path = '/airplane/samples'
    data = []
    lables = []
    
    data_m = read_data_from_dir(folder_path.format(motors_path))
    lables +=  [0*i for i in range(len(data_m))]
    data_a = read_data_from_dir(folder_path.format(airplane_path))
    lables +=  [1 + 0*i for i in range(len(data_a))]
    
    data = data_m + data_a
    
    return data, lables

if __name__ == '__main__':
    data, labels = read_data_full()

    print(len(data), len(labels))
