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
        # print(msckf_data.shape)
        msckf_timestamp = msckf_data[:,0] # for the 0th dataset
        msckf_position = np.stack(msckf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        msckf_position = msckf_position + np.array([4.688,-1.786,0.783]).reshape(1,3)
        msckf_velocity = np.stack(msckf_data[:,2])
        msckf_quat = np.stack(msckf_data[:,3]) # x,y,z,w
        return msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat

    def load_eskf(self, dataset=1):
        """Get estimate data for the eskf"""
        eskf_data = np.load(os.path.join(self.data_path, "eskf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        eskf_timestamp = eskf_data[:,0] # for the 0th dataset
        eskf_position = np.stack(eskf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        
        # print(eskf_position[0,:,:])
        # R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        R = Rotation.from_quat(np.array([-0.153,-0.8273,-0.08215,0.5341])).as_matrix()
        # R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        for iter in range(eskf_position.shape[0]):
            # np.array([4.688,-1.786,0.783])
            eskf_position[iter,:,:] = (R @ eskf_position[iter,:,:].reshape(3,1)).reshape(3,1) + np.array([4.688,-1.786,0.783]).reshape(3,1)
            # current_T = np.array([[1,0,0,eskf_position[iter,0,:]],[0,1,0,eskf_position[iter,1,:]],[0,0,1,eskf_position[iter,2,:]],[0,0,0,1]])
            # new_T = T @ current_T
            # eskf_position[iter,0,:] = new_T[0,-1]
            # eskf_position[iter,1,:] = new_T[1,-1]
            # eskf_position[iter,2,:] = new_T[2,-1]
            
        # print(eskf_position[0,:,:])

        eskf_velocity = np.stack(eskf_data[:,2])
        eskf_quat = np.stack(eskf_data[:,3]) # x,y,z,w
        return eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat

    def load_ukf(self, dataset=2):
        """Get estimate data for the eskf"""
        ukf_data = np.load(os.path.join(self.data_path, "ukf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        ukf_roll = ukf_data[0,:]
        ukf_pitch = ukf_data[1,:]
        ukf_yaw = ukf_data[2,:]
        euler_angles = np.hstack((ukf_roll.reshape(-1,1), ukf_pitch.reshape(-1,1), ukf_yaw.reshape(-1,1)))

        ukf_quat = Rotation.from_euler('XYZ', euler_angles, degrees=True).as_quat()

        _, gt_timestamp, _, _, _ = self.load_gt(dataset)
        return ukf_data, gt_timestamp ,ukf_quat

    def load_gt(self, dataset=2):
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
    load_stuff = load_data(path_euroc="/Users/aadit/Desktop/ESE650_ensemble_filtering/data/euroc_mav_dataset", path_estimate="/Users/aadit/Desktop/ESE650_ensemble_filtering/data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat = load_stuff.load_msckf(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat = load_stuff.load_eskf(dataset)
    ukf_data, ukf_timestamp ,ukf_quat = load_stuff.load_ukf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat = load_stuff.load_gt(dataset)

    # Convert quaternions to rpy
    msckf_rpy = Rotation.from_quat(msckf_quat).as_euler('XYZ', degrees=True)
    eskf_rpy = Rotation.from_quat(eskf_quat).as_euler('XYZ', degrees=True)

    # for iter in range(eskf_rpy.shape[0]):
    #     roll = eskf_rpy[iter,2]
    #     pitch = eskf_rpy[iter,1]
    #     yaw = eskf_rpy[iter,0]
    #     eskf_rpy[iter,:] = np.array([roll,pitch,yaw])

    # R = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    # R = Rotation.from_quat(np.array([-0.153,-0.8273,-0.08215,0.5341])).as_matrix()    # for MH_01
    R = Rotation.from_quat(np.array([-0.12904,-0.810903,-0.06203,0.567395])).as_matrix()    # for MH_02
    # R = np.eye(3)
    # R = Rotation.from_quat(np.array([])).as_matrix()
    # R = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    mat = Rotation.from_quat(eskf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        eskf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)

    
    ukf_rpy = Rotation.from_quat(ukf_quat).as_euler('XYZ', degrees=True)
    # R = Rotation.from_quat(np.array([-0.153,-0.8273,-0.08215,0.5341])).as_matrix()  # MH_01
    R = Rotation.from_quat(np.array([-0.12904,-0.810903,-0.06203,0.567395])).as_matrix()    # for MH_02
    # R = Rotation.from_quat(np.array([0,0,0,1])).as_matrix()

    mat = Rotation.from_quat(ukf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        ukf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)

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
        
        match_idx = []
        # print("new",len(msckf_timestamp))
        # print("new",len(eskf_timestamp))
        for i in range(len(msckf_timestamp)):
            match_idx.append(np.argmin(np.abs(eskf_timestamp - msckf_timestamp[i])))
            # print(match_idx)

        print(len(gt_idx_msckf), len(gt_timestamp[gt_idx_eskf]), len(eskf_timestamp[match_idx]))

        new_position = np.zeros_like(msckf_position)
        new_rpy = np.zeros_like(msckf_rpy)
        # print(msckf_position[:, 0].shape, eskf_position[match_idx][:, 0].shape)
        # print(((msckf_position[:, 0].reshape(2862,1) + eskf_position[match_idx][:, 0])/2).shape)
        new_position[:,0] = ((msckf_position[:, 0].reshape(2862,1) + eskf_position[match_idx][:, 0])/2).reshape(2862,)
        new_position[:,1] = ((msckf_position[:, 1].reshape(2862,1) + eskf_position[match_idx][:, 1])/2).reshape(2862,)
        new_position[:,2] = ((msckf_position[:, 2].reshape(2862,1) + eskf_position[match_idx][:, 2])/2).reshape(2862,)

        new_rpy[:,0] = ((msckf_rpy[:, 0].reshape(2862,) + eskf_rpy[match_idx][:, 0])/2).reshape(2862,)
        new_rpy[:,1] = ((msckf_rpy[:, 1].reshape(2862,) + eskf_rpy[match_idx][:, 1])/2).reshape(2862,)
        new_rpy[:,2] = ((msckf_rpy[:, 2].reshape(2862,) + eskf_rpy[match_idx][:, 2])/2).reshape(2862,)

        # plt.figure(1)
        # # plt.plot(new_position[:, 0], label="new x-pos estimate")
        # plt.plot(new_position[:, 1], label="average y-pos estimate", linestyle='dashed', color='b')
        # # plt.plot(new_position[:, 2], label="new z-pos estimate")
        # # plt.plot(gt_position[gt_idx_msckf][:, 0] - gt_position[gt_idx_msckf][0,0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_msckf][:, 1] - gt_position[gt_idx_msckf][0,1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_msckf][:, 2] - gt_position[gt_idx_msckf][0,2], label="gt z-pos", linestyle='dashdot', color='k')

        # # plt.plot(eskf_position[match_idx][:, 0], label="eskf x-pos", linestyle='dashdot')
        # plt.plot(eskf_position[match_idx][:, 1], label="eskf (baseline) y-pos", linestyle='solid', color='g')
        # # plt.plot(eskf_position[match_idx][:, 2], label="eskf z-pos", linestyle='dashdot', color='k')

        # # plt.plot(gt_position[gt_idx_msckf][:, 0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_msckf][:, 1], label="gt y-pos", linestyle='dashdot', color='k')
        # # plt.plot(gt_position[gt_idx_msckf][:, 2], label="gt z-pos", linestyle='dashdot')

        # # plt.plot(msckf_position[:, 0], label="msckf x-pos estimate")
        # plt.plot(msckf_position[:, 1], label="msckf y-pos estimate", linestyle='solid', color='r')
        # # plt.plot(msckf_position[:, 2], label="msckf z-pos estimate")

        # plt.xlabel("timestamp")
        # plt.ylabel("y position in meters")
        # plt.title("Ensemble filter estimates for y position")
        # plt.legend()



        # plt.figure(2)
        # # plt.plot(new_rpy[:, 0], label="new roll estimate")
        # # plt.plot(new_rpy[:, 1], label="average pitch estimate")
        # plt.plot(new_rpy[:, 2], label="average yaw estimate", linestyle='dashed', color='b')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 0] - gt_rpy[gt_idx_msckf][0,0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 1] - gt_rpy[gt_idx_msckf][0,1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 2] - gt_rpy[gt_idx_msckf][0,2], label="gt yaw", linestyle='dashdot', color='k')

        # # plt.plot(eskf_rpy[match_idx][:, 0], label="eskf roll", linestyle='dashdot')
        # # plt.plot(eskf_rpy[match_idx][:, 1], label="eskf (baseline) pitch", linestyle='dashdot')
        # plt.plot(eskf_rpy[match_idx][:, 2], label="eskf (baseline) yaw", linestyle='solid', color='g')

        # # plt.plot(gt_rpy[gt_idx_msckf][:, 0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 1], label="gt pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_msckf][:, 2], label="gt yaw", linestyle='dashdot', color='k')

        # # plt.plot(msckf_rpy[:, 0], label="msckf roll estimate")
        # # plt.plot(msckf_rpy[:, 1], label="msckf pitch estimate")
        # plt.plot(msckf_rpy[:, 2], label="msckf yaw estimate", linestyle='solid', color='r')

        # plt.xlabel("timestamp")
        # plt.ylabel("yaw in degrees")
        # plt.title("Ensemble filter estimates for yaw")
        # plt.legend()




        # plt.figure(1)
        # plt.plot(msckf_position[:, 0], label="msckf x-pos estimate")
        # plt.plot(msckf_position[:, 1], label="msckf y-pos estimate")
        # plt.plot(msckf_position[:, 2], label="msckf z-pos estimate")
        # # plt.plot(gt_position[gt_idx_msckf][:, 0] - gt_position[gt_idx_msckf][0,0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_msckf][:, 1] - gt_position[gt_idx_msckf][0,1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_msckf][:, 2] - gt_position[gt_idx_msckf][0,2], label="gt z-pos", linestyle='dashdot', color='k')

        # plt.plot(gt_position[gt_idx_msckf][:, 0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_msckf][:, 1], label="gt y-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_msckf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')

        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("MSCKF Position Estimate")
        # plt.legend()

        # plt.figure(2)
        # # plt.plot(msckf_velocity[:, 0], label="msckf x-vel estimate")
        # # plt.plot(msckf_velocity[:, 1], label="msckf y-vel estimate")
        # plt.plot(msckf_velocity[:, 2], label="msckf z-vel estimate")
        # # plt.plot(gt_velocity[gt_idx_msckf][:, 0] - gt_velocity[gt_idx_msckf][0,0], label="gt x-vel", linestyle='dashdot')
        # # plt.plot(gt_velocity[gt_idx_msckf][:, 1] - gt_velocity[gt_idx_msckf][0,1], label="gt y-vel", linestyle='dashdot')
        # # plt.plot(gt_velocity[gt_idx_msckf][:, 2] - gt_velocity[gt_idx_msckf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        
        # # plt.plot(gt_velocity[gt_idx_msckf][:, 0], label="gt x-vel", linestyle='dashdot')
        # # plt.plot(gt_velocity[gt_idx_msckf][:, 1], label="gt y-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_msckf][:, 2], label="gt z-vel", linestyle='dashdot', color='k')       
        
        # plt.xlabel("timestamp")
        # plt.ylabel("velocity in m/s")
        # plt.title("MSCKF Velocity Estimate")
        # plt.legend()

        # plt.figure(3)
        # # plt.plot(msckf_rpy[:, 0], label="msckf roll estimate")
        # # plt.plot(msckf_rpy[:, 1], label="msckf pitch estimate")
        # plt.plot(msckf_rpy[:, 2], label="msckf yaw estimate")
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 0] - gt_rpy[gt_idx_msckf][0,0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 1] - gt_rpy[gt_idx_msckf][0,1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 2] - gt_rpy[gt_idx_msckf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_msckf][:, 1], label="gt pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_msckf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("angle in degrees")
        # plt.title("MSCKF Orientation Estimate")
        # plt.legend()

        # plt.figure(4)
        # plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        # # plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        # # plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        # # plt.plot(gt_position[gt_idx_eskf][:, 0] - gt_position[gt_idx_eskf][0,0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 1] - gt_position[gt_idx_eskf][0,1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 2] - gt_position[gt_idx_eskf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("ESKF Position Estimate")
        # plt.legend()

        # plt.figure(5)
        # # plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        # plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        # # plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        # # plt.plot(gt_position[gt_idx_eskf][:, 0] - gt_position[gt_idx_eskf][0,0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 1] - gt_position[gt_idx_eskf][0,1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 2] - gt_position[gt_idx_eskf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        # # plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("ESKF Position Estimate")
        # plt.legend()

        # plt.figure(6)
        # # plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        # # plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        # plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        # # plt.plot(gt_position[gt_idx_eskf][:, 0] - gt_position[gt_idx_eskf][0,0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 1] - gt_position[gt_idx_eskf][0,1], label="gt y-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 2] - gt_position[gt_idx_eskf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        # # plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
        # # plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("ESKF Position Estimate")
        # plt.legend()

        # plt.figure(5)
        # plt.plot(eskf_velocity[:, 0], label="eskf x-vel estimate")
        # plt.plot(eskf_velocity[:, 1], label="eskf y-vel estimate")
        # plt.plot(eskf_velocity[:, 2], label="eskf z-vel estimate")
        # # plt.plot(gt_velocity[gt_idx_eskf][:, 0] - gt_velocity[gt_idx_eskf][0,0], label="gt x-vel", linestyle='dashdot')
        # # plt.plot(gt_velocity[gt_idx_eskf][:, 1] - gt_velocity[gt_idx_eskf][0,1], label="gt y-vel", linestyle='dashdot')
        # # plt.plot(gt_velocity[gt_idx_eskf][:, 2] - gt_velocity[gt_idx_eskf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        # plt.plot(gt_velocity[gt_idx_eskf][:, 0], label="gt x-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_eskf][:, 1], label="gt y-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_eskf][:, 2], label="gt z-vel", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("velocity in m/s")
        # plt.title("eskf Velocity Estimate")
        # plt.legend()

        # plt.figure(6)
        # plt.plot(eskf_rpy[:, 0], label="eskf roll estimate")
        # # plt.plot(eskf_rpy[:, 1], label="eskf pitch estimate")
        # # plt.plot(eskf_rpy[:, 2], label="eskf yaw estimate")
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 0] - gt_rpy[gt_idx_eskf][0,0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 1] - gt_rpy[gt_idx_eskf][0,1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 2] - gt_rpy[gt_idx_eskf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        # plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("angle in degrees")
        # plt.title("ESKF Orientation Estimate")
        # plt.legend()

        # plt.figure(7)
        # # plt.plot(eskf_rpy[:, 0], label="eskf roll estimate")
        # plt.plot(eskf_rpy[:, 1], label="eskf pitch estimate")
        # # plt.plot(eskf_rpy[:, 2], label="eskf yaw estimate")
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 0] - gt_rpy[gt_idx_eskf][0,0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 1] - gt_rpy[gt_idx_eskf][0,1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 2] - gt_rpy[gt_idx_eskf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("angle in degrees")
        # plt.title("ESKF Orientation Estimate")
        # plt.legend()

        # plt.figure(8)
        # # plt.plot(eskf_rpy[:, 0], label="eskf roll estimate")
        # # plt.plot(eskf_rpy[:, 1], label="eskf pitch estimate")
        # plt.plot(eskf_rpy[:, 2], label="eskf yaw estimate")
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 0] - gt_rpy[gt_idx_eskf][0,0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 1] - gt_rpy[gt_idx_eskf][0,1], label="gt pitch", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 2] - gt_rpy[gt_idx_eskf][0,2], label="gt yaw", linestyle='dashdot', color='k')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot')
        # # plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("angle in degrees")
        # plt.title("ESKF Orientation Estimate")
        # plt.legend()

        plt.figure(9)
        plt.plot(ukf_rpy[:,0], label="ukf roll estimate")
        # plt.plot(ukf_rpy[:,1], label="ukf pitch estimate")
        # plt.plot(ukf_rpy[:,2], label="ukf yaw estimate")
        plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        # plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("UKF Orientation Estimate")
        plt.legend()

        plt.figure(10)
        # plt.plot(ukf_rpy[:,0], label="ukf roll estimate")
        plt.plot(ukf_rpy[:,1], label="ukf pitch estimate")
        # plt.plot(ukf_rpy[:,2], label="ukf yaw estimate")
        # plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("UKF Orientation Estimate")
        plt.legend()

        plt.figure(11)
        # plt.plot(ukf_rpy[:,0], label="ukf roll estimate")
        # plt.plot(ukf_rpy[:,1], label="ukf pitch estimate")
        plt.plot(ukf_rpy[:,2], label="ukf yaw estimate")
        # plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        # plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("UKF Orientation Estimate")
        plt.legend()

        

        plt.show()

