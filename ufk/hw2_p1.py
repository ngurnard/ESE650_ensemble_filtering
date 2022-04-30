import numpy as np
import matplotlib.pyplot as plt


class EKF:
    """
    Class for dataset creation and estimation value of constant a
    """
    def __init__(self):
        """
        Constructor for class. 
        """

        # Variance of dynamic noise and observation noise
        self.R = 1
        self.Q = 0.5

        # Initial estimate of the mean and vairance of state x
        self.initial_mean = 1
        self.initial_variance = 2
        self.x_0 = np.random.normal(self.initial_mean, np.sqrt(self.initial_variance))

        # True value of a
        self.true_a = -1
        # Initial estimate of a during estimation task
        self.initial_a = -10
        # Noise incorporated in calculation of a during estimation task
        self.P = 0.01
        # Generated dataset of observations 
        self.dataset = []
    
    def dataset_create(self, iterations):
        """
        Function to create the dataset of observations based on true value of a
        """

        x = np.random.normal(self.initial_mean, np.sqrt(self.initial_variance))
        for i in range(iterations):
            y = np.sqrt(((x**2) + 1)) + np.random.normal(0, np.sqrt(self.Q))
            self.dataset.append(y)
            x = (self.true_a * x) + np.random.normal(0, np.sqrt(self.R)) 
            
    
    def propagate(self, mu_k_k, sigma_k_k):
        """
        Function to perform propagation step for EFK
        """

        A = np.array([[mu_k_k[1,0], mu_k_k[0,0]],[0,1]])
        R_cov = np.array([[self.R, 0],[0, self.P]])

        mu_prop = np.array([[mu_k_k[1,0]*mu_k_k[0,0]],[mu_k_k[1,0]]])
        sigma_prop = (A @ sigma_k_k @ A.T) + R_cov

        return mu_prop, sigma_prop

    def update(self, mu_prop, sigma_prop, y_next):
        """
        Function to perform Update step for EKF
        """

        C = np.array([[((mu_prop[0,0])/(np.sqrt(((mu_prop[0,0]**2)+1)))), 0]])
        K = (sigma_prop @ C.T) @ np.linalg.inv((C @ sigma_prop @ C.T) + self.Q)

        mu_update = mu_prop + (K * (y_next - np.sqrt(((mu_prop[0,0]**2)+1))))
        sigma_update = (np.eye(2) - K @ C) @ sigma_prop

        return mu_update, sigma_update

if __name__ == '__main__':
    """
    Main function of script
    """

    # Creating object for EKF class
    ekf = EKF()
    # Setting number of iterations for dataset creation.
    num_iterations = 100
    ekf.dataset_create(num_iterations)

    # Setting initial estimates of mean and variance of combined state of system (z)
    mu_k_k = np.array([[ekf.initial_mean],[ekf.initial_a]])
    sigma_k_k = np.array([[ekf.initial_variance, 0],[0, 2]])

    positive_dev = []
    negative_dev = []
    mean = []

    # Looping through all observations for estimating value of a
    for y_next in ekf.dataset:
        mu_prop, sigma_prop = ekf.propagate(mu_k_k, sigma_k_k)
        mu_update, sigma_update = ekf.update(mu_prop, sigma_prop, y_next)

        mu_k_k = mu_update.copy()
        sigma_k_k = sigma_update.copy()

        positive_dev.append(mu_k_k[1,0] + np.sqrt(sigma_k_k[1,1]))
        negative_dev.append(mu_k_k[1,0] - np.sqrt(sigma_k_k[1,1]))
        mean.append(mu_k_k[1,0])
    
    # Plotting the true value and estimated value of a
    iterations = np.arange(0, num_iterations, 1)
    a = np.array([ekf.true_a] * num_iterations)
    plt.plot(iterations, positive_dev, negative_dev)
    plt.plot(iterations, a, mean)
    plt.legend(('Positive deviation from mean','Negative deviation from mean','True a', 'Mean value of a'))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Values of a')
    plt.show()