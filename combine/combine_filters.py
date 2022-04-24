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
        # print(gt_quat)
        gt_quat = np.roll(gt_quat, -1, axis = 1) 
        # print(gt_quat)
        gt_velocity = np.stack(gt_data[:,8:11])

        return gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat


if __name__ == "__main__":

    ### Define parameters
    dataset = 1
    show_plots = True # specify if you want plots to be shown
    match_timesteps = True # if you want the plots to only display pts where the timestamps match

    ### Initlialize
    load_stuff = load_data(path_euroc="./data/euroc_mav_dataset", path_estimate="./data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat = load_stuff.load_msckf(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat = load_stuff.load_eskf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat = load_stuff.load_gt(dataset)

    # Convert quaternions to rpy
    msckf_rpy = Rotation.from_quat(msckf_quat).as_euler('XYZ', degrees=True)
    eskf_rpy = Rotation.from_quat(eskf_quat).as_euler('XYZ', degrees=True)
    gt_rpy = Rotation.from_quat(gt_quat).as_euler('XYZ', degrees=True)

    if (match_timesteps == False):
        ### Plot the data as is
        plt.figure(1)
        plt.plot(msckf_position[:, 0], label="msckf x-pos estimate", color="r")
        plt.plot(msckf_position[:, 1], label="msckf y-pos estimate", color="g")
        plt.plot(msckf_position[:, 2], label="msckf z-pos estimate", color="b")
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("MSCKF Position Estimate")
        plt.legend()
        # plt.show()

        plt.figure(2)
        plt.plot(eskf_position[:, 0], label="eskf x-pos estimate", color="r")
        plt.plot(eskf_position[:, 1], label="eskf y-pos estimate", color="g")
        plt.plot(eskf_position[:, 2], label="eskf z-pos estimate", color="b")
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("ESKF Position Estimate")
        plt.legend()

        plt.figure(3)
        plt.plot(gt_position[:, 0] - gt_position[0,0], label="gt x-pos estimate", color="r")
        plt.plot(gt_position[:, 1] - gt_position[0,1], label="gt y-pos estimate", color="g")
        plt.plot(gt_position[:, 2] - gt_position[0,2], label="gt z-pos estimate", color="b")
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("Ground Truth Estimate")
        plt.legend()
        plt.show()
    else:
        # msckf_idx = np.where(msckf_timestamp == gt_timestamp) # none of these match the gt??
        # eskf_idx = np.where(eskf_timestamp == gt_timestamp)
        # print(msckf_timestamp)
        # print(gt_timestamp)
        # print(eskf_timestamp)
        # print(msckf_idx, eskf_timestamp.shape)

        gt_idx_msckf = []
        for i in range(len(msckf_timestamp)):
            gt_idx_msckf.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i])))

        gt_idx_eskf = []
        for i in range(len(eskf_timestamp)):
            gt_idx_eskf.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

        print(len(gt_idx_msckf), len(gt_timestamp[gt_idx_msckf]))

        plt.figure(1)
        plt.plot(msckf_position[:, 0], label="msckf x-pos estimate")
        plt.plot(msckf_position[:, 1], label="msckf y-pos estimate")
        plt.plot(msckf_position[:, 2], label="msckf z-pos estimate")
        plt.plot(gt_position[gt_idx_msckf][:, 0] - gt_position[gt_idx_msckf][0,0], label="gt x-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_msckf][:, 1] - gt_position[gt_idx_msckf][0,1], label="gt y-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_msckf][:, 2] - gt_position[gt_idx_msckf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("MSCKF Position Estimate")
        plt.legend()

        plt.figure(2)
        plt.plot(msckf_velocity[:, 0], label="msckf x-vel estimate")
        plt.plot(msckf_velocity[:, 1], label="msckf y-vel estimate")
        plt.plot(msckf_velocity[:, 2], label="msckf z-vel estimate")
        plt.plot(gt_velocity[gt_idx_msckf][:, 0] - gt_velocity[gt_idx_msckf][0,0], label="gt x-vel", linestyle='dashdot')
        plt.plot(gt_velocity[gt_idx_msckf][:, 1] - gt_velocity[gt_idx_msckf][0,1], label="gt y-vel", linestyle='dashdot')
        plt.plot(gt_velocity[gt_idx_msckf][:, 2] - gt_velocity[gt_idx_msckf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("velocity in m/s")
        plt.title("MSCKF Velocity Estimate")
        plt.legend()

        plt.figure(3)
        plt.plot(msckf_rpy[:, 0], label="msckf roll estimate")
        plt.plot(msckf_rpy[:, 1], label="msckf pitch estimate")
        plt.plot(msckf_rpy[:, 2], label="msckf yaw estimate")
        plt.plot(gt_rpy[gt_idx_msckf][:, 0] - gt_rpy[gt_idx_msckf][0,0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 1] - gt_rpy[gt_idx_msckf][0,1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 2] - gt_rpy[gt_idx_msckf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("MSCKF Orientation Estimate")
        plt.legend()

        plt.figure(4)
        plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        plt.plot(gt_position[gt_idx_eskf][:, 0] - gt_position[gt_idx_eskf][0,0], label="gt x-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_eskf][:, 1] - gt_position[gt_idx_eskf][0,1], label="gt y-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_eskf][:, 2] - gt_position[gt_idx_eskf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("ESKF Position Estimate")
        plt.legend()

        plt.figure(5)
        plt.plot(eskf_velocity[:, 0], label="eskf x-vel estimate")
        plt.plot(eskf_velocity[:, 1], label="eskf y-vel estimate")
        plt.plot(eskf_velocity[:, 2], label="eskf z-vel estimate")
        plt.plot(gt_velocity[gt_idx_eskf][:, 0] - gt_velocity[gt_idx_eskf][0,0], label="gt x-vel", linestyle='dashdot')
        plt.plot(gt_velocity[gt_idx_eskf][:, 1] - gt_velocity[gt_idx_eskf][0,1], label="gt y-vel", linestyle='dashdot')
        plt.plot(gt_velocity[gt_idx_eskf][:, 2] - gt_velocity[gt_idx_eskf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("velocity in m/s")
        plt.title("eskf Velocity Estimate")
        plt.legend()

        plt.figure(6)
        plt.plot(eskf_rpy[:, 0], label="eskf roll estimate")
        plt.plot(eskf_rpy[:, 1], label="eskf pitch estimate")
        plt.plot(eskf_rpy[:, 2], label="eskf yaw estimate")
        plt.plot(gt_rpy[gt_idx_eskf][:, 0] - gt_rpy[gt_idx_eskf][0,0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_eskf][:, 1] - gt_rpy[gt_idx_eskf][0,1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_eskf][:, 2] - gt_rpy[gt_idx_eskf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("ESKF Orientation Estimate")
        plt.legend()

        plt.show()

