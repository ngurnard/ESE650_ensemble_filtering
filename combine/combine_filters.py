# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import pandas as pd
import os

class load_data():
    """
    A class used to load in data that can be used accross any file we make.
    """
    def __init__(self, path_euroc, path_estimate):
        """
        Inputs: 
        1) path_euroc = the path to the parent folder of the Euroc dataset
        2) path_estimate = the path to the folder where the state estimate data is located
        """
        ### load data from the correct directory
        self.data_path = path_estimate
        self.gt_path = path_euroc # the ground truth path

    def load_msckf(self, dataset=1):
        """Get estimate data for the msckf"""
        msckf_data = np.load(os.path.join(self.data_path, "msckf_data" + str(dataset) + ".npy" ), allow_pickle=True)
        msckf_timestamp = msckf_data[:,0] # for the 0th dataset
        msckf_position = np.stack(msckf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        msckf_velocity = np.stack(msckf_data[:,2])
        msckf_quat = np.stack(msckf_data[:,3]) # x,y,z,w
        return msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat

    def load_eskf(self, dataset=1):
        """Get estimate data for the eskf"""
        eskf_data = np.load(os.path.join(self.data_path, "eskf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        eskf_timestamp = eskf_data[:,0] # for the 0th dataset
        eskf_position = np.stack(eskf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        eskf_velocity = np.stack(eskf_data[:,2])
        eskf_quat = np.stack(eskf_data[:,3]) # x,y,z,w
        return eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat

    def load_gt(self, dataset=1):
        """Get ground truth data for the specified dataset"""
        gt_path2 = self.gt_path + "/MH_0" + str(dataset) # specify the directory
        if dataset == 1 or dataset == 2:
            gt_path2 = gt_path2 + "_easy"
        elif dataset == 3:
            gt_path2 = gt_path2 + "_medium"
        else:
            gt_path2 = gt_path2 + "_difficult"

        print(gt_path2)
        print(os.path.join(gt_path2, "/mav0/state_groundtruth_estimate0/data.csv"))
        gt_data = np.loadtxt(os.path.join(gt_path2, "mav0/state_groundtruth_estimate0/data.csv"), delimiter=",")
        gt_timestamp = gt_data[:,0] # for the 0th dataset
        gt_position = np.stack(gt_data[:,1:4]) # the np.stack() converts the array to arrays to 2d array to actually use
        gt_quat = np.stack(gt_data[:,4:8]) # w,x,y,z
        gt_velocity = np.stack(gt_data[:,8:11])

        return gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat


if __name__ == "__main__":

    ### Define parameters
    dataset = 1
    show_plots = True # specify if you want plots to be shown

    ### Initlialize
    load_stuff = load_data(path_euroc="./data/euroc_mav_dataset", path_estimate="./data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat = load_stuff.load_msckf(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat = load_stuff.load_eskf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat = load_stuff.load_gt(dataset)

    ### Plot the data as is
    plt.figure(1)
    plt.plot(msckf_position[:, 0], label="msckf x-pos estimate")
    plt.plot(msckf_position[:, 1], label="msckf y-pos estimate")
    plt.plot(msckf_position[:, 2], label="msckf z-pos estimate")
    plt.xlabel("timestamp")
    plt.ylabel("position in meters")
    plt.title("MSCKF Position Estimate")
    # plt.show()

    plt.figure(2)
    plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
    plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
    plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
    plt.xlabel("timestamp")
    plt.ylabel("position in meters")
    plt.title("ESKF Position Estimate")

    plt.figure(3)
    plt.plot(gt_position[:, 0] - gt_position[0,0], label="gt x-pos estimate")
    plt.plot(gt_position[:, 1] - gt_position[0,1], label="gt y-pos estimate")
    plt.plot(gt_position[:, 2] - gt_position[0,2], label="gt z-pos estimate")
    plt.xlabel("timestamp")
    plt.ylabel("position in meters")
    plt.title("Ground Truth Estimate")
    plt.show()