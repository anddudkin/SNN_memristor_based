import os

import h5py
import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
import requests
import scipy.constants as const
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.stats import linregress


def load_SiO_x_multistate() -> np.ndarray:
    """Load SiO_x data from multiple conductance states.

    Returns:
        Array of shape `(2, num_states, num_points)`. The first dimension
            combines current and voltage values.
    """
    path = os.path.join(_create_and_get_data_dir(), "SiO_x-multistate-data.mat")
    _validate_data_path(path, url="https://zenodo.org/record/5762184/files/excelDataCombined.mat")
    data = loadmat(path)["data"]
    data = np.flip(data, axis=2)
    data = np.transpose(data, (1, 2, 0))
    data = data[:2, :, :]
    return data
