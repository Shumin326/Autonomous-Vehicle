# Navigation with SLAM

Perform SLAM for moving vehicles or robots to navigate unknwon environments.

The goal is to get the trajectories of the robot and obtain corresponding maps of the test building.

 - The whole SLAM process can be divided into three parts: mapping, prediction and update. Before everything, note that we use particle:(position x, position y, orientation theta) to represent the current state. 
    - The mapping part transform the obstacles(wall-shaped) obtained from Lidar data into a map
    - The prediction part uses only odometry information to predict the next state. 
    - The update part combines the current map with particles(find correlation) and use that to update the particle weights.

Results:

![](results/0.gif)
![](results/1.gif)
![](results/0.png)

