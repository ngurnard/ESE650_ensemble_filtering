# %% Imports
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from complementary_filter import complementary_filter_update
import pdb

# %%  CSV imu file
start = time.time()
fname = './data/euroc_mav_dataset/MH_01_easy/mav0/imu0/data.csv'

# %%
imu0 = np.genfromtxt(fname, delimiter=',', dtype='float64', skip_header=1)

# %% pull out components of data set - different views of matrix

# timestamps in nanoseconds
t = imu0[:, 0]
print(t.shape)

# angular velocities in radians per second
angular_velocity = imu0[:, 1:4]

# linear acceleration in meters per second^2
linear_acceleration = imu0[:, 4:]

# gyro_bias = gt_data[:,11:14]
# acc_bias = gt_data[:,14:]
# pdb.set_trace()
# %% Process the imu data

n = imu0.shape[0]

euler = np.zeros((n, 3))

R_correction = Rotation.from_quat(np.array([-0.153, -0.8273, -0.08215, 0.5341]))
R = Rotation.identity()
for i in range(1, n):
    # print(i)
    dt = (t[i] - t[i - 1]) * 1e-9
    R = complementary_filter_update(R, angular_velocity[i - 1] - np.array([-0.00317, 0.021267, 0.078502]), linear_acceleration[i] - np.array([-0.025266, 0.136696, 0.075593]), dt)
    new_R = R_correction*R
    euler[i] = new_R.as_euler('XYZ', degrees=True)

# %% Plots

t2 = (t - t[0]) * 1e-9

end = time.time()
print("runtime: ", (end - start))

# pdb.set_trace()
np.save('./data/filter_outputs/complementary_data1', np.concatenate((t.reshape((-1,1)), euler), axis = 1))

fig = plt.figure()
plt.plot(t2, euler[:, 0], 'b', label='yaw')
plt.plot(t2, euler[:, 1], 'g', label='pitch')
plt.plot(t2, euler[:, 2], 'r', label='roll')
plt.ylabel('degrees')
plt.xlabel('seconds')
plt.title('Attitude of Quad')
plt.legend()
plt.savefig('full_data_result.png')
plt.show()
