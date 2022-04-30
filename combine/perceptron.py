## Import the necessary libraries
import torch
import numpy as np
import matplotlib

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


def pos_perceptron(x_vec, y_vec, z_vec):
    """
    A function that takes in a vector for x, y, z pos and
    returns the output of a single layer percetron

    Inputs:
        x_vec: a (n, m) array of x position filter estimates where n is the number of timesteps, m is the number of filters
        y_vec: a (n, m) array of y position filter estimates where n is the number of timesteps, m is the number of filters
        z_vec: a (n, m) array of z position filter estimates where n is the number of timesteps, m is the number of filters

    Outputs:
        x:a  (n, 1) vector that has a new estimate for x (output of perceptron)
        y: a (n, 1) vector that has a new estimate for y (output of perceptron)
        z: a (n, 1) vector that has a new estimate for z (output of perceptron) 
    """

    x_model = x_net() # initialize the neural net object
    print(x_model)



if __name__ == "__main__":
    print("This file is just for calling the perceptron in combine_filters.py")