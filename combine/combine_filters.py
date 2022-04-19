# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import pandas as pd
import os

# load data from the correct directory
data_path = './data/filter_outputs'

# Load in data
msckf_data = np.load(os.path.join(data_path, "msckf.npy"))
# print(msckf_data)
eskf_data = np.load(os.path.join(data_path, "eskf_data_v2.npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
print(eskf_data.shape, eskf_data[0])

if __name__ == "__main__":
    print("\n\n\n Done")