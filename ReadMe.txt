The training data file contains two variables. The variable "F_train" contains:

* F_train.t - They are the times of the observation (N x 1 matrix)
* F_train.f - They are the feature values (N x 42 matrix)
* F_train.valid - A flag telling you if there were too mnay missing values and hence the features are not good (N x 1 matrix)

The features were computed over a 1 second window with 75% overlap, and they are:
*  1: 6 - Mean (of Accelerometer and Gyro values)
*  7:12 - Variance (of Accelerometer and Gyro values)
* 13:18 - Skewness (of Accelerometer and Gyro values)
* 19:24 - Kurtosis (of Accelerometer and Gyro values)
* 25:39 - Cross Correlation (between Accelerometer and Gyro)
* 40:42 - Gyro Energy

The variable "gtLabel_train" contains the ground truth labels. They represent the modes: (0) Stationary, (1) moving in the middle of the arena, (2) moving CW on the boundary, and (3) moving CCW on the boundary.
