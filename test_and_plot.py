# Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import click
import torch

from train import *
from load_data import load_data
import os

# Get user input if they want
@click.command()
@click.option('--dataset', default=1, help='specify the machine hall dataset number. Valid datasets in range [1,5]', type=int)
@click.option('--ensemble', default=True, help='specify which model to use as a boolean: simple average (False) or perceptron (True)', type=bool)

def main(dataset, ensemble):
    # Run python main.py --help to see how to provide command line arguments
    # Check if the user input makes sense
    if not dataset in [1, 2, 3, 4, 5]:
        raise ValueError('Unknown argument --data %s'%dataset)
    if not ensemble in [True, False]:
        raise ValueError('Unknown argument --ensemble %s'%ensemble)

    ### Define parameters
    # dataset = 1
    match_timesteps = True # if you want the plots to only display pts where the timestamps match. Set to False if you want to debug individual filters
    # perceptron = True # if you want to combine the outputs with a perceptron. Otherwise shows a simple average comparison

    ### Initlialize
    load_stuff = load_data(path_euroc=os.getcwd() + "/data/euroc_mav_dataset", path_estimate=os.getcwd() + "/data/filter_outputs") # initilize the load_data object

    ### Get the data we need
    ukf_data, ukf_timestamp, ukf_rpy = load_stuff.load_ukf(dataset)
    gt_data, gt_timestamp, gt_position, gt_velocity, gt_rpy = load_stuff.load_gt(dataset)
    eskf_data, eskf_timestamp, eskf_position, eskf_velocity, eskf_rpy = load_stuff.load_eskf(dataset)
    # msckf_data, msckf_timestamp, msckf_position, msckf_velocity, msckf_rpy = load_stuff.load_msckf(dataset)
    complementary_data, complementary_timestamp, complementary_rpy = load_stuff.load_complementary(dataset)
    
    ## Perceptron Code -----------------------------------------------------------------
    if (ensemble == True):
    
        ## ROLL ----------------------------------------------------------------
        # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
        match_idx = []
        for i in range(len(eskf_timestamp)):
            match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
        # Make a numpy array of all of the filters roll
        x_or_array = np.vstack((eskf_rpy[:, 0], complementary_rpy[match_idx][:, 0], ukf_rpy[match_idx][:, 0], gt_rpy[match_idx][:, 0])).transpose()
       
        # load in the trained model AFTER running perceptron.py
        x_model = x_net()
        x_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/x_model.pt'))
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
        y_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/y_model.pt'))
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
        z_model.load_state_dict(torch.load(os.getcwd() + '/data/trained_models/z_model.pt'))
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

        # Show all of the Plots
        plt.show()

    ## PLOTTING CODE FOR DUBUGGING -------------------------------------------------------------------
    if (match_timesteps == False and ensemble == False):
        ### Plot the data as is
        plt.figure(1)
        plt.plot(msckf_position[:, 0], label="msckf x-pos estimate", color="r")
        plt.plot(msckf_position[:, 1], label="msckf y-pos estimate", color="g")
        plt.plot(msckf_position[:, 2], label="msckf z-pos estimate", color="b")
        plt.xlabel("timestamp")
        plt.ylabel("position in meters")
        plt.title("MSCKF Position Estimate")
        plt.legend()

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

    elif (match_timesteps == True and ensemble == False):
        """
        This is to compare filters against the ground truth and also to see the simply average comparison
        """
        
        ## Match timesteps for plotting
        # Ground truth with MSCKF
        gt_idx_msckf = []
        for i in range(len(msckf_timestamp)):
            gt_idx_msckf.append(np.argmin(np.abs(gt_timestamp - msckf_timestamp[i])))
        # Ground Truth with ESKF
        gt_idx_eskf = []
        for i in range(len(eskf_timestamp)):
            gt_idx_eskf.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
         # Ground Truth and Complementary
        gt_idx_complementary = []
        for i in range(len(complementary_timestamp)):
            gt_idx_complementary.append(np.argmin(np.abs(gt_timestamp - complementary_timestamp[i])))
        # Ground Truth with UKF
        gt_idx_ukf = []
        for i in range(len(ukf_timestamp)):
            gt_idx_ukf.append(np.argmin(np.abs(gt_timestamp - ukf_timestamp[i])))
        # ESKF with MSCKF
        match_idx = []
        for i in range(len(msckf_timestamp)):
            match_idx.append(np.argmin(np.abs(eskf_timestamp - msckf_timestamp[i])))
        # Complementary with MSCKF ( this is orientation only)
        match_idx2 = []
        for i in range(len(msckf_timestamp)):
            match_idx2.append(np.argmin(np.abs(complementary_timestamp - msckf_timestamp[i])))
       

        # Get the simple average of the MSCKF and ESKF output for when the timestamps match
        # new_position = np.zeros_like(msckf_position)
        # new_rpy = np.zeros_like(msckf_rpy)
        # new_position[:,0] = ((msckf_position[:, 0] + eskf_position[match_idx][:, 0])/2).reshape(-1,) # divide by 2 for the simple average
        # new_position[:,1] = ((msckf_position[:, 1] + eskf_position[match_idx][:, 1])/2).reshape(-1,)
        # new_position[:,2] = ((msckf_position[:, 2] + eskf_position[match_idx][:, 2])/2).reshape(-1,)
        # new_rpy[:,0] = ((msckf_rpy[:, 0] + eskf_rpy[match_idx][:, 0])/2)
        # new_rpy[:,1] = ((msckf_rpy[:, 1] + eskf_rpy[match_idx][:, 1])/2)
        # new_rpy[:,2] = ((msckf_rpy[:, 2] + eskf_rpy[match_idx][:, 2])/2)
        # plt.figure(1)
        # plt.plot(msckf_position[:, 0], label="msckf x-pos estimate")
        # plt.plot(msckf_position[:, 1], label="msckf y-pos estimate")
        # plt.plot(msckf_position[:, 2], label="msckf z-pos estimate")
        # plt.plot(gt_position[gt_idx_msckf][:, 0] - gt_position[gt_idx_msckf][0,0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_msckf][:, 1] - gt_position[gt_idx_msckf][0,1], label="gt y-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_msckf][:, 2] - gt_position[gt_idx_msckf][0,2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("MSCKF Position Estimate")
        # plt.legend()

        # plt.figure(2)
        # plt.plot(msckf_velocity[:, 0], label="msckf x-vel estimate")
        # plt.plot(msckf_velocity[:, 1], label="msckf y-vel estimate")
        # plt.plot(msckf_velocity[:, 2], label="msckf z-vel estimate")
        # plt.plot(gt_velocity[gt_idx_msckf][:, 0] - gt_velocity[gt_idx_msckf][0,0], label="gt x-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_msckf][:, 1] - gt_velocity[gt_idx_msckf][0,1], label="gt y-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_msckf][:, 2] - gt_velocity[gt_idx_msckf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("velocity in m/s")
        # plt.title("MSCKF Velocity Estimate")
        # plt.legend()

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
        # plt.plot(gt_position[gt_idx_eskf][:, 0], label="gt x-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 1], label="gt y-pos", linestyle='dashdot')
        # plt.plot(gt_position[gt_idx_eskf][:, 2], label="gt z-pos", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("position in meters")
        # plt.title("ESKF Position Estimate")
        # plt.legend()

        # plt.figure(5)
        # plt.plot(eskf_velocity[:, 0], label="eskf x-vel estimate")
        # plt.plot(eskf_velocity[:, 1], label="eskf y-vel estimate")
        # plt.plot(eskf_velocity[:, 2], label="eskf z-vel estimate")
        # plt.plot(gt_velocity[gt_idx_eskf][:, 0] - gt_velocity[gt_idx_eskf][0,0], label="gt x-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_eskf][:, 1] - gt_velocity[gt_idx_eskf][0,1], label="gt y-vel", linestyle='dashdot')
        # plt.plot(gt_velocity[gt_idx_eskf][:, 2] - gt_velocity[gt_idx_eskf][0,2], label="gt z-vel", linestyle='dashdot', color='k')
        # plt.xlabel("timestamp")
        # plt.ylabel("velocity in m/s")
        # plt.title("eskf Velocity Estimate")
        # plt.legend()

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
        plt.title("UKF Orientation Estimate")
        plt.legend()

        plt.figure(8)
        plt.plot(complementary_rpy[:, 0], label="complementary roll estimate")
        plt.plot(complementary_rpy[:, 1], label="complementary pitch estimate")
        plt.plot(complementary_rpy[:, 2], label="complementary yaw estimate")
        plt.plot(gt_rpy[:, 0], label="gt roll", linestyle='dashdot')
        plt.plot(gt_rpy[:, 1], label="gt pitch", linestyle='dashdot')
        plt.plot(gt_rpy[:, 2], label="gt yaw", linestyle='dashdot', color='k')
        plt.xlabel("timestamp")
        plt.ylabel("angle in degrees")
        plt.title("Complementary Orientation Estimate")
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
        avg_rpy = np.zeros_like(eskf_rpy)
        avg_rpy[:,0] = ((eskf_rpy[:, 0] + ukf_rpy[gt_idx_eskf][:, 0] + complementary_rpy[gt_idx_eskf][:, 0])/3)
        avg_rpy[:,1] = ((eskf_rpy[:, 1] + ukf_rpy[gt_idx_eskf][:, 1] + complementary_rpy[gt_idx_eskf][:, 1])/3)
        avg_rpy[:,2] = ((eskf_rpy[:, 2] + ukf_rpy[gt_idx_eskf][:, 2] + complementary_rpy[gt_idx_eskf][:, 2])/3)

        plt.figure(9)
        plt.plot(avg_rpy[:, 0], label="average roll estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 0], label="eskf roll estimate", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 0], label="gt roll", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 0], label="ukf roll estimate", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 0], label="complementary roll estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("roll estimate in degrees")
        plt.title("Ensemble Filter Estimates for Roll - Simple Average")
        plt.legend()
        ukf_loss_roll = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 0] - ukf_rpy[gt_idx_eskf][:, 0])))
        eskf_loss_roll = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 0] - eskf_rpy[:, 0])))
        complementary_loss_roll = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 0] - complementary_rpy[gt_idx_eskf][:, 0])))
        avg_loss_roll = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 0] - avg_rpy[:, 0])))
        print(f"ukf roll loss: {ukf_loss_roll}, eskf roll loss: {eskf_loss_roll}, complimentary roll loss: {complementary_loss_roll}, model output roll loss: {avg_loss_roll}")

        plt.figure(10)
        plt.plot(avg_rpy[:, 1], label="average pitch estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 1], label="eskf pitch", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt pitch", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 1], label="ukf pitch", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 1], label="complementary pitch estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("pitch estimate in degrees")
        plt.title("Ensemble Filter Estimates for Pitch - Simple Average")
        plt.legend()
        ukf_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - ukf_rpy[gt_idx_eskf][:, 1])))
        eskf_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - eskf_rpy[:, 1])))
        complementary_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - complementary_rpy[gt_idx_eskf][:, 1])))
        avg_loss_pitch = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - avg_rpy[:, 1])))
        print(f"ukf pitch loss: {ukf_loss_pitch}, eskf pitch loss: {eskf_loss_pitch}, complimentary pitch loss: {complementary_loss_pitch}, model output pitch loss: {avg_loss_pitch}")

        plt.figure(11)
        plt.plot(avg_rpy[:, 1], label="average yaw estimate", linestyle='dashed', color='b')
        plt.plot(eskf_rpy[:, 1], label="eskf yaw", linestyle='solid', color='g')
        plt.plot(gt_rpy[gt_idx_eskf][:, 1], label="gt yaw", linestyle='dashdot', color='k')
        plt.plot(ukf_rpy[gt_idx_eskf][:, 1], label="ukf yaw", linestyle='dashdot', color='m')
        plt.plot(complementary_rpy[gt_idx_eskf][:, 1], label="complementary yaw estimate", linestyle='solid', color='r')
        plt.xlabel("timestamp")
        plt.ylabel("yaw estimate in degrees")
        plt.title("Ensemble Filter Estimates for Yaw - Simple Average")
        plt.legend()
        ukf_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - ukf_rpy[gt_idx_eskf][:, 1])))
        eskf_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - eskf_rpy[:, 1])))
        complementary_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - complementary_rpy[gt_idx_eskf][:, 1])))
        avg_loss_yaw = np.sum(np.abs((gt_rpy[gt_idx_eskf][:, 1] - avg_rpy[:, 1])))
        print(f"ukf yaw loss: {ukf_loss_yaw}, eskf yaw loss: {eskf_loss_yaw}, complimentary yaw loss: {complementary_loss_yaw}, model output yaw loss: {avg_loss_yaw}")

        plt.show()

if __name__ == "__main__":    
    print("\nRun python main.py --help to see how to provide command line arguments\n\n")
    main()
    