# Panda_Robot_Sim
A simulation environment for the Franka emika Panda robot with Bullet Physics engine

In order to use this repository, you must first install the Bullet Physics engine locally, you can do that by downloading the Bullet3 repository and following the process listed there:
https://github.com/bulletphysics/bullet3

This repository is based on the "panda_sim.py" file that is located in the Bullet3 examples folder.
A few methods have been added by me in order to make the use of this simulation simple and easy for all kinds of research purposes and for collecting datasets for oflline RL.

After installing the Bullet engine, download this repository and run "Panda_sim_example.py" to see the simulation in action.

## Environment
This simulation is built to mimic the Mujoco Kitchen environment in PyBullet for those who cant get a Mujoco license, this includes a model of a corner kitchen, a table, the Panda robot and a few Objects.
Kitchen model includes:
* Kitchen counter
* Two shelves
* Cabinets (upper cabinet doors can turn or slide open)
* Valves, Two of which can turn 90 degrees (the first and forth)

![Image of the simulation environment](https://github.com/BenyaHalevy/Panda_Robot_Sim/blob/master/images/Kitchen_env.JPG)

## How to use the simulation for data collection
### Manual control
For trying and testing movement manually while the simulation is running, first run "GUI_for_sim.py" to run the simulation with a graphical user interface for controlling the Panda robot. The control inteface shows all joint angles and the gripper state in real time and allows the user to set the robots position and orientation in the simulation axis and open or close the gripper. **important - y axis is the height axis, while x and z are the horizontal plane axes**. The user can also "reset" the simulation anytime in order to reload the simulation and the robot initial position.

![Image of the user interface](https://github.com/BenyaHalevy/Panda_Robot_Sim/blob/master/images/user_interface.JPG)

This tool can help try out different trajectories and robot states to use for automatic dataset collection for offline RL.

### Automatic data collection
After testing out different robot states with the user interface, open "State_class_rdc.py" and set the states you wish to use for collecting a reach objective dataset. The states must include the position and orientation of the robot in that state and also all joint angles and gripper state (note that most positions and orientations of the robot can be achived with a number of joint angle combinations).

After setting the states, open "reach_data_collect.py", set the expiriment parameters that match the number of states and size of dataset you wish to collect and run the script. This will automaticly choose combinations of two states from the state list and run the amount of tests set for reaching from the 1st state to the 2nd. While running the simulated tests, the (s,a,s',r) tuple will be saved in a dedicated .csv file for each state combination. The agent will only recive a reward of **"1"** if it succeeds in reaching the 2nd state within a normally distributed tolerance chosen in each test episode. 

When the program is done collecting the datasets, a legend .csv file will be saved containing information about all state combinations that were tested and their respective success rates. **You can find datasets that were allready collected by me in the datasets directory.** 
