## Import the necessary libraries
from random import shuffle
from turtle import color
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm
import pdb
from combine_filters import *
from scipy.spatial.transform import Rotation

## Define the classes for each of the perceptrons
class x_net(torch.nn.Module):
    """
    A single layer perceptron for x position
    """
    def __init__(self):
        super(x_net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

class y_net(torch.nn.Module):
    """
    A single layer perceptron for x position
    """
    def __init__(self):
        super(y_net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

class z_net(torch.nn.Module):
    """
    A single layer perceptron for x position
    """
    def __init__(self):
        super(z_net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=1) # this is a fully connected layer (single layer perceptron)

    def forward(self, x):
        x = self.fc1(x) # this is the forward pass for the fully connected layer
        return x

def orientation_perceptron(x_arr, y_arr, z_arr):

    # ## ROLL --------------------------------------------
    # np.random.shuffle(x_arr)
    # x_train, x_test, x_labels_train, x_labels_test = train_test_split(x_arr[:, :-1], x_arr[:, -1:], test_size=0.20, random_state=42)
    # x_train = torch.from_numpy(x_train) # convert the numpy array to a tensor
    # x_test = torch.from_numpy(x_test) # convert the numpy array to a tensor
    # x_labels_train = torch.from_numpy(x_labels_train) # convert the numpy array to a tensor
    # x_labels_test = torch.from_numpy(x_labels_test) # convert the numpy array to a tensor
    # train_tds = torch.utils.data.TensorDataset(x_train, x_labels_train)
    # test_tds = torch.utils.data.TensorDataset(x_test, x_labels_test)
    # x_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=16, shuffle=False)
    # x_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    # x_model = x_net()
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(x_model.parameters(), lr=1e-2, weight_decay=1e-4) 
    # train_mse = []
    # epochs = 2000
    # # training iterations
    # for epoch in tqdm.tqdm(range(epochs)):
    #     running_loss = 0
    #     count = 0
    #     for itr, (image, label) in enumerate(x_trainloader):
    #         # zero gradient
    #         optimizer.zero_grad()
    #         # forward path
    #         y_predicted = x_model(image.float())
    #         loss = criterion(y_predicted, label.float())
    #         # if(itr == 0):
    #         #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
    #         running_loss += loss.item()
    #         # backpropagating
    #         loss.backward()
    #         # optimizes the weights
    #         optimizer.step()
    #         count += 1
    #     train_mse.append(running_loss/(count))
    #     # if (epoch+1) % 3 == 0:
    #     #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    # x_ls = []
    # x_pred = []
    # with torch.no_grad():
    #     for itr, (image, label) in enumerate(x_testloader):
    #         x_predicted = x_model(image.float())
    #         loss = criterion(x_predicted, label.float())
    #         x_ls.append(label.item())
    #         x_pred.append(x_predicted.item())
    #     print(f'MSE loss of test is {loss:.4f}')

    # # torch.save(x_model.state_dict(), './combine/x_model.pt')

    # plt.figure(1)
    # plt.plot(train_mse)
    # plt.title('Training Loss')

    # plt.figure(2)
    # plt.plot(x_ls, color = 'g')
    # plt.plot(x_pred, color = 'r')
    # plt.title('Prediction')
    # plt.show()


    # ## PITCH --------------------------------------------
    # np.random.shuffle(y_arr)
    # y_train, y_test, y_labels_train, y_labels_test = train_test_split(y_arr[:, :-1], y_arr[:, -1:], test_size=0.20, random_state=42)
    # y_train = torch.from_numpy(y_train) # convert the numpy array to a tensor
    # y_test = torch.from_numpy(y_test) # convert the numpy array to a tensor
    # y_labels_train = torch.from_numpy(y_labels_train) # convert the numpy array to a tensor
    # y_labels_test = torch.from_numpy(y_labels_test) # convert the numpy array to a tensor
    # train_tds = torch.utils.data.TensorDataset(y_train, y_labels_train)
    # test_tds = torch.utils.data.TensorDataset(y_test, y_labels_test)
    # y_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=16, shuffle=False)
    # y_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    # y_model = y_net()
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(y_model.parameters(), lr=1e-2, weight_decay=1e-4) 
    # train_mse = []
    # epochs = 2000
    # # training iterations
    # for epoch in tqdm.tqdm(range(epochs)):
    #     running_loss = 0
    #     count = 0
    #     for itr, (image, label) in enumerate(y_trainloader):
    #         # zero gradient
    #         optimizer.zero_grad()
    #         # forward path
    #         y_predicted = y_model(image.float())
    #         loss = criterion(y_predicted, label.float())
    #         # if(itr == 0):
    #         #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
    #         running_loss += loss.item()
    #         # backpropagating
    #         loss.backward()
    #         # optimizes the weights
    #         optimizer.step()
    #         count += 1
    #     train_mse.append(running_loss/(count))
    #     # if (epoch+1) % 3 == 0:
    #     #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    # y_ls = []
    # y_pred = []
    # with torch.no_grad():
    #     for itr, (image, label) in enumerate(y_testloader):
    #         y_predicted = y_model(image.float())
    #         loss = criterion(y_predicted, label.float())
    #         y_ls.append(label.item())
    #         y_pred.append(y_predicted.item())
    #     print(f'MSE loss of test is {loss:.4f}')

    # # torch.save(y_model.state_dict(), './combine/y_model.pt')

    # plt.figure(1)
    # plt.plot(train_mse)
    # plt.title('Training Loss')

    # plt.figure(2)
    # plt.plot(y_ls, color = 'g')
    # plt.plot(y_pred, color = 'r')
    # plt.title('Prediction')
    # plt.show()

    ## YAW --------------------------------------------
    np.random.shuffle(z_arr)
    z_train, z_test, z_labels_train, z_labels_test = train_test_split(z_arr[:, :-1], z_arr[:, -1:], test_size=0.20, random_state=42)
    z_train = torch.from_numpy(z_train) # convert the numpy array to a tensor
    z_test = torch.from_numpy(z_test) # convert the numpy array to a tensor
    z_labels_train = torch.from_numpy(z_labels_train) # convert the numpy array to a tensor
    z_labels_test = torch.from_numpy(z_labels_test) # convert the numpy array to a tensor
    train_tds = torch.utils.data.TensorDataset(z_train, z_labels_train)
    test_tds = torch.utils.data.TensorDataset(z_test, z_labels_test)
    z_trainloader = torch.utils.data.DataLoader(train_tds, batch_size=16, shuffle=False)
    z_testloader = torch.utils.data.DataLoader(test_tds, shuffle=False)

    z_model = z_net()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(z_model.parameters(), lr=5e-3, weight_decay=1e-4) 
    train_mse = []
    epochs = 5000
    # training iterations
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        count = 0
        for itr, (image, label) in enumerate(z_trainloader):
            # zero gradient
            optimizer.zero_grad()
            # forward path
            z_predicted = z_model(image.float())
            loss = criterion(z_predicted, label.float())
            # if(itr == 0):
            #     print(f'epoch: {epoch+1}, batch: {itr+1}, loss: {loss.item():.4f}')
            running_loss += loss.item()
            # backpropagating
            loss.backward()
            # optimizes the weights
            optimizer.step()
            count += 1
        train_mse.append(running_loss/(count))
        # if (epoch+1) % 3 == 0:
        #     print(f'epoch: {epoch+1}, loss: {running_loss:.4f}')

    z_ls = []
    z_pred = []
    with torch.no_grad():
        for itr, (image, label) in enumerate(z_testloader):
            z_predicted = z_model(image.float())
            loss = criterion(z_predicted, label.float())
            z_ls.append(label.item())
            z_pred.append(z_predicted.item())
        print(f'MSE loss of test is {loss:.4f}')

    torch.save(z_model.state_dict(), './combine/z_model.pt')

    plt.figure(1)
    plt.plot(train_mse)
    plt.title('Training Loss')

    plt.figure(2)
    plt.plot(z_ls, color = 'g')
    plt.plot(z_pred, color = 'r')
    plt.title('Prediction')
    plt.show()



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
    R = Rotation.from_quat(np.array([-0.153,-0.8273,-0.08215,0.5341])).as_matrix() # from world frame to vicon world frame
    mat = Rotation.from_quat(eskf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        eskf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)
    ukf_rpy = Rotation.from_quat(ukf_quat).as_euler('XYZ', degrees=True)
    R = Rotation.from_quat(np.array([-0.153,-0.8273,-0.08215,0.5341])).as_matrix() # from world frame to vicon world frame
    mat = Rotation.from_quat(ukf_quat).as_matrix()
    for iter in range(mat.shape[0]):
        mat[iter,:,:] = R @ mat[iter,:,:]
        ukf_rpy[iter,:] = Rotation.from_matrix(mat[iter,:,:]).as_euler('XYZ', degrees=True)


    # Match the timesteps of the ESKF with gt in order to make a perceptron of the positions
    match_idx = []
    for i in range(len(eskf_timestamp)):
        match_idx.append(np.argmin(np.abs(gt_timestamp - eskf_timestamp[i])))
    
    # Make a numpy array of all of the filters x,y,z positions (THIS IS NOW BROKEN)
    # x_pos_array = np.vstack((msckf_position[:, 0], eskf_position[match_idx][:, 0])).transpose()
    # y_pos_array = np.vstack((msckf_position[:, 1], eskf_position[match_idx][:, 1])).transpose()
    # z_pos_array = np.vstack((msckf_position[:, 2], eskf_position[match_idx][:, 2])).transpose()
    # print("TESTING 1: ", x_pos_array.shape, y_pos_array.shape, z_pos_array.shape)
    # Make a numpy array of all of the filters x,y,z positions
    x_or_array = np.vstack((eskf_rpy[:, 0], complementary_rpy[match_idx][:, 0], ukf_rpy[match_idx][:, 0], gt_rpy[match_idx][:, 0])).transpose()
    y_or_array = np.vstack((eskf_rpy[:, 1], complementary_rpy[match_idx][:, 1], ukf_rpy[match_idx][:, 1], gt_rpy[match_idx][:, 1])).transpose()
    z_or_array = np.vstack((eskf_rpy[:, 2], complementary_rpy[match_idx][:, 2], ukf_rpy[match_idx][:, 2], gt_rpy[match_idx][:, 2])).transpose()

    # Pass it into the perceptron!
    # pos_perceptron(x_pos_array, y_pos_array, z_pos_array)
    orientation_perceptron(x_or_array, y_or_array, z_or_array) # UNCOMMENT IN ORDER TO TRAIN THE PERCEPTRON
    # Model class must be defined somewhere