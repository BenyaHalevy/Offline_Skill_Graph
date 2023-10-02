import PySimpleGUI as sg
from multiprocessing import Process
import sys
import pybullet as pb
import pybullet_data as pd
import math
import time
import numpy as np
from panda_robot_sim.resources import panda_Kitchen_sim as panda_sim

deg2rad = math.pi/180
rad2deg = 180/math.pi
toggle = 0

''' Loading the Panda Simulation '''
pb.connect(pb.GUI)
pb.configureDebugVisualizer(pb.COV_ENABLE_Y_AXIS_UP, 1)
pb.setAdditionalSearchPath(pd.getDataPath())
timeStep = 1. / 60.
pb.setTimeStep(timeStep)
pb.setGravity(0, -9.8, 0)
panda = panda_sim.PandaSim(pb, [0, 0, 0])
cube = pb.loadURDF("cube_small.urdf", np.array([0.45, 0.1, 0.0]), [-0.5, -0.5, -0.5, 0.5])
Px, Py, Pz = panda.start_pos
Ax, Ay, Az = np.multiply(rad2deg, panda.start_orn)
grip = int(panda.grip_closed)
jointPoses = panda.get_joint_poses()
pb.resetJointState(panda.kitchen, jointIndex=10, targetValue=0.28)
''' Loading the Control Panel GUI '''
sg.theme('DarkAmber')
layout = [[sg.Text('ROBOT STATE:')],
          [sg.Text('J1:'), sg.Text(jointPoses[0], size=(7,1), key='-J1-'), sg.Text('J2:'), sg.Text(jointPoses[1], size=(7,1), key='-J2-'), sg.Text('J3:'), sg.Text(jointPoses[2], size=(7,1), key='-J3-'), sg.Text('J4:'), sg.Text(jointPoses[3], size=(7,1), key='-J4-')],
          [sg.Text('J5:'), sg.Text(jointPoses[4], size=(7,1), key='-J5-'), sg.Text('J6:'), sg.Text(jointPoses[5], size=(7,1), key='-J6-'), sg.Text('J7:'), sg.Text(jointPoses[6], size=(7,1), key='-J7-'), sg.Text('Gripper:'), sg.Text(grip, size=(7,1), key='-Gripper-')],
          [sg.Text('')],
          [sg.Text('SET POSE AND ORIENTATION:')],
          [sg.Text('Px:'), sg.InputText(Px, size=(9,1), key='-Px-'), sg.Text('Py:'), sg.InputText(Py, size=(9,1), key='-Py-'), sg.Text('Pz:'), sg.InputText(Pz, size=(9,1), key='-Pz-')],
          [sg.Text('Ax:'), sg.InputText(Ax, size=(9,1), key='-Ax-'), sg.Text('Ay:'), sg.InputText(Ay, size=(9,1), key='-Ay-'), sg.Text('Az:'), sg.InputText(Az, size=(9,1), key='-Az-')],
          [sg.Text('\nClose/Open Gripper')],
          [sg.Text('Gripper:'), sg.InputText(grip, size=(9,1), key='-Set Gripper-')],
          [sg.Text('')],
          [sg.Button('SET'), sg.Button('Exit'), sg.Button('Reset', key='-RESET-')]]


# Create the Window
window = sg.Window('Panda Sim GUI', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=1)
    if event == sg.WIN_CLOSED or event == 'Exit':  # if user closes window or clicks cancel
        break
    elif event == 'SET':
        ''' Get values From GUI '''
        Px = float(values['-Px-'])
        Py = float(values['-Py-'])
        Pz = float(values['-Pz-'])
        Ax = float(values['-Ax-'])
        Ay = float(values['-Ay-'])
        Az = float(values['-Az-'])
        grip = int(values['-Set Gripper-'])
        # print(jointPoses[9:])

    elif event == '-RESET-':
        panda.bullet_client.resetSimulation()
        pb.setGravity(0, -9.8, 0)
        panda = panda_sim.PandaSim(pb, [0, 0, 0])
        cube = pb.loadURDF("cube_small.urdf", np.array([0.45, 0.1, 0.0]), [-0.5, -0.5, -0.5, 0.5])
        pb.resetJointState(panda.kitchen, jointIndex=10, targetValue=0.28)
        Px, Py, Pz = panda.start_pos
        Ax, Ay, Az = np.multiply(rad2deg, panda.start_orn)

    ''' Set Robot Pose and Orientation '''
    panda.inverse_kinematics([Px, Py, Pz], np.multiply(deg2rad, [Ax, Ay, Az]))
    if grip == 1:
        panda.close_gripper()
    else:
        panda.open_gripper()

    ''' Get Robot State '''
    grip = int(panda.grip_command)
    jointPoses = panda.get_joint_poses()

    ''' Update GUI Control Panel Indicators'''
    window['-J1-'].update(str(jointPoses[0]))
    window['-J2-'].update(str(jointPoses[1]))
    window['-J3-'].update(str(jointPoses[2]))
    window['-J4-'].update(str(jointPoses[3]))
    window['-J5-'].update(str(jointPoses[4]))
    window['-J6-'].update(str(jointPoses[5]))
    window['-J7-'].update(str(jointPoses[6]))
    window['-Gripper-'].update(str(grip))

    pb.stepSimulation()
    time.sleep(timeStep)

    closed = panda.get_gripper_state()
    # if closed == 1.0:
    #     print(closed)

    # print(panda.get_grippers_distance())
    rads = np.multiply(deg2rad, [Ax, Ay, Az])
    # print(panda.bullet_client.getQuaternionFromEuler(rads))

window.close()
