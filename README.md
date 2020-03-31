# SLAM
SLAM for THOR humanoid robot. This is an individual course homework project for ESE 650: Learning in Robotics at UPenn.

We're given three datasets collected by IMU, odometry and Lidar sensors that are attached onto the THOR robot:

The goal is to get the trajectories of the robot and obtain corresponding maps of the test building.

 - The whole SLAM process can be divided into three parts: mapping, prediction and update. Before everything, note that we use particle:(position x, position y, orientation theta) to represent the current state. 
    - The mapping part transform the obstacles obtained from Lidar data(polar coordinates) into cartesian x-y coordinates and transform it from THOR robot's body frame into world frame. 
    - The prediction part uses only odometry information to predict the next state. 
    - The update part combines the current map with particles(find correlation) and use that to update the particle weights.

Results:

![](results/0.gif)
![](results/1.gif)
![](results/0.png)

