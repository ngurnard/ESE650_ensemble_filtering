# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import pandas as pd
import os

import torch

from perceptron import *

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
        """Get estimate data for the msckf - position and orientation"""
        msckf_data = np.load(os.path.join(self.data_path, "msckf_data" + str(dataset) + ".npy" ), allow_pickle=True)
        msckf_timestamp = msckf_data[:,0] # for the 0th dataset
        msckf_position = np.stack(msckf_data[:,1]) # the np.stack() converts the array to arrays to 2d array to actually use
        msckf_velocity = np.stack(msckf_data[:,2])
        msckf_quat = np.stack(msckf_data[:,3]) # x,y,z,w
        return msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat

    # def load_biases(self, dataset=1):
    #     """Get bias estimates that were learned in the MSCKF Algorithm. These will be used as the true bias values"""
    #     gyro_bias = np.load(os.path.join(self.data_path, "msckf_gyro_bias" + str(dataset) + ".npy" ), allow_pickle=True)
    #     acc_bias = np.load(os.path.join(self.data_path, "msckf_acc_bias" + str(dataset) + ".npy" ), allow_pickle=True)
    #     return gyro_bias, acc_bias

    def load_eskf(self, dataset=1):
        """Get estimate data for the eskf - position and orientation"""
        eskf_data = np.load(os.path.join(self.data_path, "eskf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        eskf_timestamp = eskf_data[:,0] # for the 0th dataset
        eskf_position = np.stack(eskf_data[:,1]).reshape(-1,3) # the np.stack() converts the array to arrays to 2d array to actually use
        eskf_velocity = np.stack(eskf_data[:,2]).reshape(-1,3)
        eskf_quat = np.stack(eskf_data[:,3]) # x,y,z,w
        # print("TESTING ESKF: ", eskf_position.shape, eskf_velocity.shape, eskf_quat.shape)

        # Convert the reading from the filter world frame (first frame is the world frame) to the vicon world frame (position)
        if (dataset == 1):
            initial_quat = np.array([-0.153,-0.8273,-0.08215,0.5341])
            initial_pos = np.array([4.688,-1.786,0.783]).transpose()
        R = Rotation.from_quat(initial_quat).as_matrix() # this is the first quaternion from the GROUND TRUTH DATA
        for iter in range(eskf_position.shape[0]):
            eskf_position[iter,:] = (R @ eskf_position[iter,:].transpose()) + initial_pos

        return eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat

    def load_complementary(self, dataset = 1):
        """Get estimate data for the complementary filter - orientation only"""
        complementary_data = np.load(os.path.join(self.data_path, "complementary_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        complementary_timestamp = complementary_data[:,0]
        complementary_euler = np.stack(complementary_data[:,1:])

        return complementary_data, complementary_timestamp, complementary_euler

    def load_ukf(self, dataset=1):
        """Get estimate data for the ukf - orientation only (for now)"""
        ukf_data = np.load(os.path.join(self.data_path, "ukf_data" + str(dataset) + ".npy"), allow_pickle=True) # each state timestep, position, velocity, quaternion
        ukf_roll = ukf_data[0,:]
        ukf_pitch = ukf_data[1,:]
        ukf_yaw = ukf_data[2,:]
        euler_angles = np.hstack((ukf_roll.reshape(-1,1), ukf_pitch.reshape(-1,1), ukf_yaw.reshape(-1,1)))

        ukf_quat = Rotation.from_euler('xyz', euler_angles, degrees=True).as_quat()

        _, gt_timestamp, _, _, _ = self.load_gt(dataset)
        return ukf_data, gt_timestamp, ukf_quat

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
    perceptron = True # if you want to combine the outputs with a perceptron

    ### Initlialize
    load_stuff = load_data(path_euroc="./data/euroc_mav_dataset", path_estimate="./data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_quat = load_stuff.load_msckf(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_quat = load_stuff.load_eskf(dataset)
    ukf_data, ukf_timestamp ,ukf_quat = load_stuff.load_ukf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_quat = load_stuff.load_gt(dataset)
    complementary_data, complementary_timestamp, complementary_euler = load_stuff.load_complementary(dataset)

    # Convert quaternions to rpy
    msckf_rpy = Rotation.from_quat(msckf_quat).as_euler('XYZ', degrees=True)
    eskf_rpy = Rotation.from_quat(eskf_quat).as_euler('XYZ', degrees=True)
    gt_rpy = Rotation.from_quat(gt_quat).as_euler('XYZ', degrees=True)
    complementary_rpy = complementary_euler

    # Convert the reading from the filter world frame (first frame is the world frame) to the vicon world frame (rotation)
    if (dataset == 1):
        initial_quat = np.array([-0.153,-0.8273,-0.08215,0.5341])
    R = Rotation.from_quat(initial_quat).as_matrix() # from world frame to vicon world frame
    mat = Rotation.from_quat(eskf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        eskf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)
    ukf_rpy = Rotation.from_quat(ukf_quat).as_euler('XYZ', degrees=True)
    R = Rotation.from_quat(initial_quat).as_matrix() # from world frame to vicon world frame
    mat = Rotation.from_quat(ukf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        ukf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)

    
    ## Perceptron Code -----------------------------------------------------------------
    if (perceptron == True):

        ## Match timesteps for plotting
        # Keep the following for reference!
        """
        # Ground truth with MSCKF
        gt_idx1 = []
        for i in range(len(msckf_timestamp)):
            gt_idx1.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i]))) # match the INDICES of where the timestamps match
        gt_idx1 = np.array(gt_idx1)
        # print("gt_idx1: ", gt_idx1.shape, gt_idx1)
        gt_timestamp2 = gt_timestamp[gt_idx1]
        # Output of last with ESKF
        gt_idx2 = []
        for i in range(len(eskf_timestamp)):
            gt_idx2.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        gt_idx2 = np.array(gt_idx2)
        # print("gt_idx2: ", gt_idx2.shape, gt_idx2)
        idx = np.intersect1d(gt_idx1, gt_idx2) # these are the indices in which the timestamps match for all 3 sets
        # print("idx: ", idx)
        """

        ## ROLL ----------------------------------------------------------------
        # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        # Make a numpy array of all of the filters roll
        x_or_array = np.vstack((eskf_rpy[:, 0], complementary_rpy[match_idx][:, 0], ukf_rpy[match_idx][:, 0], gt_rpy[match_idx][:, 0])).transpose()
       
        # load in the trained model AFTER running perceptron.py
        x_model = x_net()
        x_model.load_state_dict(torch.load('./combine/x_model' + str(dataset) + '.pt'))
        x_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        x_test, x_labels_test = x_or_array[:, :-1], x_or_array[:, -1:]
        x_test = torch.from_numpy(x_test) # convert the numpy array to a tensor
        x_labels_test = torch.from_numpy(x_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(x_test, x_labels_test)
        x_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        x_ls = []
        x_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(x_testloader):
                x_predicted = x_model(image.float())
                loss = criterion(x_predicted, label.float())
                x_ls.append(label.item())
                x_pred.append(x_predicted.item())
            print(f'MSE loss of test is {loss:.4f}')      

        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

        plt.figure(1)
        plt.plot(x_pred, label="network roll estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 0], label="eskf roll", linestyle='solid', color='g')
        plt.plot(gt_rpy[match_idx][:, 0], label="gt roll", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 0], label="ukf roll", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[match_idx][:, 0], label="complementary roll estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("roll estimate in degrees")
        plt.title("Ensemble Filter Estimates for Roll - Network Output")
        plt.legend()

        ukf_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - ukf_rpy[match_idx][:, 0])))
        eskf_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - eskf_rpy[:, 0])))
        complementary_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - complementary_rpy[match_idx][:, 0])))
        new_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - x_pred)))
        print(f"ukf roll loss: {ukf_loss_roll}, eskf roll loss: {eskf_loss_roll}, complimentary roll loss: {complementary_loss_roll}, model output roll loss: {new_loss_roll}")


        ## PITCH ----------------------------------------------------------------
        # Make a numpy array of all of the filters pitch
        y_or_array = np.vstack((eskf_rpy[:, 1], complementary_rpy[match_idx][:, 1], ukf_rpy[match_idx][:, 1], gt_rpy[match_idx][:, 1])).transpose()

        # load in the trained model AFTER running perceptron.py
        y_model = y_net()
        y_model.load_state_dict(torch.load('./combine/y_model' + str(dataset) + '.pt'))
        y_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        y_test, y_labels_test = y_or_array[:, :-1], y_or_array[:, -1:]
        y_test = torch.from_numpy(y_test) # convert the numpy array to a tensor
        y_labels_test = torch.from_numpy(y_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(y_test, y_labels_test)
        y_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        y_ls = []
        y_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(y_testloader):
                y_predicted = y_model(image.float())
                loss = criterion(y_predicted, label.float())
                y_ls.append(label.item())
                y_pred.append(y_predicted.item())
            print(f'MSE loss of test is {loss:.4f}')
       
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

        plt.figure(2)
        plt.plot(y_pred, label="network pitch estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 1], label="eskf pitch", linestyle='solid', color='g')
        plt.plot(gt_rpy[match_idx][:, 1], label="gt pitch", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 1], label="ukf pitch", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[match_idx][:, 1], label="complementary pitch estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("pitch estimate in degrees")
        plt.title("Ensemble Filter Estimates for Pitch - Network Output")
        plt.legend()
        

        ukf_loss_pitch = np.sum(np.abs((gt_rpy[match_idx][:, 1] - ukf_rpy[match_idx][:, 1])))
        eskf_loss_pitch = np.sum(np.abs((gt_rpy[match_idx][:, 1] - eskf_rpy[:, 1])))
        complementary_loss_pitch = np.sum(np.abs((gt_rpy[match_idx][:, 1] - complementary_rpy[match_idx][:, 1])))
        new_loss_pitch = np.sum(np.abs((gt_rpy[match_idx][:, 1] - y_pred)))
        print(f"ukf pitch loss: {ukf_loss_pitch}, eskf pitch loss: {eskf_loss_pitch}, complimentary pitch loss: {complementary_loss_pitch}, model output pitch loss: {new_loss_pitch}")

        ## YAW ----------------------------------------------------------------
        # Make a numpy array of all of the filters yaw
        z_or_array = np.vstack((eskf_rpy[:, 2], complementary_rpy[match_idx][:, 2], ukf_rpy[match_idx][:, 2], gt_rpy[match_idx][:, 2])).transpose()

        # load in the trained model AFTER running perceptron.py
        z_model = z_net()
        z_model.load_state_dict(torch.load('./combine/z_model' + str(dataset) + '.pt'))
        z_model.eval() # dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

        z_test, z_labels_test = z_or_array[:, :-1], z_or_array[:, -1:]
        z_test = torch.from_numpy(z_test) # convert the numpy array to a tensor
        z_labels_test = torch.from_numpy(z_labels_test) # convert the numpy array to a tensor
        test_tds = torch.utils.data.TensorDataset(z_test, z_labels_test)
        z_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)
        criterion = torch.nn.MSELoss()
        z_ls = []
        z_pred = []
        with torch.no_grad():
            for itr, (image, label) in enumerate(z_testloader):
                z_predicted = z_model(image.float())
                loss = criterion(z_predicted, label.float())
                z_ls.append(label.item())
                z_pred.append(z_predicted.item())
            print(f'MSE loss of test is {loss:.4f}')
       
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

        plt.figure(3)
        plt.plot(z_pred, label="network yaw estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 2], label="eskf yaw", linestyle='solid', color='g')
        plt.plot(gt_rpy[match_idx][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 2], label="ukf yaw", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[match_idx][:, 2], label="complementary yaw estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("yaw estimate in degrees")
        plt.title("Ensemble Filter Estimates for Yaw - Network Output")
        plt.legend()
        

        ukf_loss_yaw = np.sum(np.abs((gt_rpy[match_idx][:, 2] - ukf_rpy[match_idx][:, 2])))
        eskf_loss_yaw = np.sum(np.abs((gt_rpy[match_idx][:, 2] - eskf_rpy[:, 2])))
        complementary_loss_yaw = np.sum(np.abs((gt_rpy[match_idx][:, 2] - complementary_rpy[match_idx][:, 2])))
        new_loss_yaw = np.sum(np.abs((gt_rpy[match_idx][:, 2] - z_pred)))
        print(f"ukf yaw loss: {ukf_loss_yaw}, eskf yaw loss: {eskf_loss_yaw}, complimentary yaw loss: {complementary_loss_yaw}, model output yaw loss: {new_loss_yaw}")


        plt.show()
    ## PLOTTING CODE -------------------------------------------------------------------
    if (match_timesteps == False and show_plots == True and perceptron == False):
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

        plt.figure(4)
        plt.plot(gt_rpy[:, 0] - gt_rpy[0,0], label="gt x-pos estimate", color="r")
        plt.plot(gt_rpy[:, 1] - gt_rpy[0,1], label="gt y-pos estimate", color="g")
        plt.plot(gt_rpy[:, 2] - gt_rpy[0,2], label="gt z-pos estimate", color="b")
        plt.xlabel("timestamp")
        plt.ylabel("rpy in meters")
        plt.title("Ground Truth Estimate")
        plt.legend()
        plt.show()

    elif (show_plots == True and match_timesteps == True and perceptron == False):
        
        ## Match timesteps for plotting
        # Ground truth with MSCKF
        gt_idx_msckf = []
        for i in range(len(msckf_timestamp)):
            gt_idx_msckf.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i])))
        # Ground Truth with ESKF
        gt_idx_eskf = []
        for i in range(len(eskf_timestamp)):
            gt_idx_eskf.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        # ESKF with MSCKF
        match_idx = []
        for i in range(len(msckf_timestamp)):
            match_idx.append(np.argmin(np.abs(eskf_timestamp - msckf_timestamp[i])))
            # print(match_idx)
        # Complementary with MSCKF ( this is orientation only)
        match_idx2 = []
        for i in range(len(msckf_timestamp)):
            match_idx2.append(np.argmin(np.abs(complementary_timestamp - msckf_timestamp[i])))
            # print(match_idx)
        # Ground truth and complementary
        gt_idx_complementary = []
        for i in range(len(complementary_timestamp)):
            gt_idx_complementary.append(np.argmin(np.abs(gt_timestamp - complementary_timestamp[i])))
        # Ground Truth with ESKF
        gt_idx_ukf = []
        for i in range(len(ukf_timestamp)):
            gt_idx_ukf.append(np.argmin(np.abs(gt_timestamp - ukf_timestamp[i])))
        # print(len(gt_idx_msckf), len(gt_timestamp[gt_idx_msckf]))

        # Get the simple average of the MSCKF and ESKF output for when the timestamps match
        new_position = np.zeros_like(msckf_position)
        new_rpy = np.zeros_like(msckf_rpy)
        new_position[:,0] = ((msckf_position[:, 0] + eskf_position[match_idx][:, 0])/2).reshape(-1,) # divide by 2 for the simple average
        new_position[:,1] = ((msckf_position[:, 1] + eskf_position[match_idx][:, 1])/2).reshape(-1,)
        new_position[:,2] = ((msckf_position[:, 2] + eskf_position[match_idx][:, 2])/2).reshape(-1,)
        new_rpy[:,0] = ((msckf_rpy[:, 0] + eskf_rpy[match_idx][:, 0])/2)
        new_rpy[:,1] = ((msckf_rpy[:, 1] + eskf_rpy[match_idx][:, 1])/2)
        new_rpy[:,2] = ((msckf_rpy[:, 2] + eskf_rpy[match_idx][:, 2])/2)

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
        plt.plot(gt_rpy[gt_idx_msckf][:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_msckf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("MSCKF Orientation Estimate")
        plt.legend()

        # plt.figure(4)
        # plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        # plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        # plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        # plt.plot(gt_position[gt_idx_eskf][:, 0] - gt_position[gt_idx_eskf][0,0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 1] - gt_position[gt_idx_eskf][0,1], label="gt y-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 2] - gt_position[gt_idx_eskf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("ESKF Position Estimate")
        # plt.legend()

        plt.figure(4)
        plt.plot(eskf_position[:, 0], label="eskf x-pos estimate")
        plt.plot(eskf_position[:, 1], label="eskf y-pos estimate")
        plt.plot(eskf_position[:, 2], label="eskf z-pos estimate")
        plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
        plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
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
        plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[gt_idx_eskf][:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("ESKF Orientation Estimate")
        plt.legend()

        plt.figure(7)
        plt.plot(ukf_rpy[:, 0], label="ukf roll estimate")
        plt.plot(ukf_rpy[:, 1], label="ukf pitch estimate")
        plt.plot(ukf_rpy[:, 2], label="ukf yaw estimate")
        plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("ukf Orientation Estimate")
        plt.legend()

        ## Simple Average Plot -- presentaion
        # plt.figure(7)
        # plt.plot(new_position[:, 1], label="average y-pos estimate", linestyle='dashed', color='b')
        # plt.plot(eskf_position[match_idx][:, 1], label="eskf (baseline) y-pos", linestyle='solid', color='g')
        # plt.plot(gt_position[gt_idx_msckf][:, 1], label="gt y-pos", linestyle='dashdot', color='k')
        # plt.plot(msckf_position[:, 1], label="msckf y-pos estimate", linestyle='solid', color='r')
        # plt.xlabel("timestamp")
        # plt.ylabel("y position in meters")
        # plt.title("Ensemble filter estimates for y position - Simple Average")
        # plt.legend()
        # plt.figure(8)
        # plt.plot(complementary_rpy[:,0], label = 'comp_yaw')
        # plt.plot(complementary_rpy[:,1], label = 'comp_pitch')
        # plt.plot(complementary_rpy[:,2], label = 'comp_roll')
        # plt.plot(gt_rpy[gt_idx_complementary][:, 0], label="gt_yaw", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_complementary][:, 1], label="gt_pitch", linestyle='dashdot')
        # plt.plot(gt_rpy[gt_idx_complementary][:, 2], label="gt_roll", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("angle in degrees")
        # plt.title("Complementary Orientation Estimate")
        # plt.legend()

        ## Simple Averaging - All Filters
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))

        new_rpy = np.zeros_like(eskf_rpy)
        new_rpy[:,0] = ((eskf_rpy[:, 0] + ukf_rpy[match_idx][:, 0] + complementary_rpy[match_idx][:, 0])/3)
        new_rpy[:,1] = ((eskf_rpy[:, 1] + ukf_rpy[match_idx][:, 1] + complementary_rpy[match_idx][:, 1])/3)
        new_rpy[:,2] = ((eskf_rpy[:, 2] + ukf_rpy[match_idx][:, 2] + complementary_rpy[match_idx][:, 2])/3)
        plt.figure(8)
        plt.plot(new_rpy[:, 0], label="average y-rot estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 0], label="eskf (baseline) y-rot", linestyle='solid', color='g')
        plt.plot(gt_rpy[match_idx][:, 0], label="gt y-rot", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[match_idx][:, 0], label="ukf y-rot", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[match_idx][:, 0], label="complementary y-rot estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("y position in meters")
        plt.title("Ensemble filter estimates for y position - Simple Average")
        plt.legend()

        ukf_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - ukf_rpy[match_idx][:, 0])))
        eskf_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - eskf_rpy[:, 0])))
        complementary_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - complementary_rpy[match_idx][:, 0])))
        new_loss_roll = np.sum(np.abs((gt_rpy[match_idx][:, 0] - new_rpy[:, 0])))
        print(ukf_loss_roll, eskf_loss_roll, complementary_loss_roll, new_loss_roll)

        plt.show()

