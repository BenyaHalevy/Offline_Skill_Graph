from multiprocessing import Process
import sys
import pybullet as pb
import pybullet_data as pd
import math
import time
import numpy as np
from panda_robot_sim.resources import panda_Kitchen_sim as panda_sim
from pynput import keyboard

deg2rad = math.pi/180


def simulate(pb, panda):
    counter = 0
    state = 0
    while True:
        if panda.check_in_position() and state == 0:
            panda.inverse_kinematics([0.5, 0.65, 0.03], np.multiply(deg2rad, [90, 0, 90]))
            if panda.check_in_position():
                state = 1
        elif state == 1:
            panda.inverse_kinematics([0.55, 0.65, 0.03], np.multiply(deg2rad, [90, 0, 90]))
            if panda.check_in_position():
                state = 2
        elif state == 2:
            panda.close_gripper()
            counter = counter + 1
            if counter == 50:
                state = 3
                counter = 0
        elif state == 3:
            panda.inverse_kinematics([0.55, 0.65, 0.25], np.multiply(deg2rad, [90, 0, 90]))
            counter = counter + 1
            if counter > 50:
                state = 4
                counter = 0
        elif state == 4:
            panda.open_gripper()
            counter = counter + 1
            if counter == 50:
                state = 5
                counter = 0
        elif state == 5:
            panda.inverse_kinematics([0.4, 0.65, 0.25], np.multiply(deg2rad, [95, 0, 90]))
            if panda.check_in_position():
                state = 6
        elif state == 6:
            panda.inverse_kinematics([0.3, 0.5, -0.3], np.multiply(deg2rad, [90, 0, 90]))
            if panda.check_in_position():
                state = 7
        elif state == 7:
            panda.inverse_kinematics([0.3, 0.5, -0.3], np.multiply(deg2rad, [90, 0, 0]))
            if panda.check_in_position():
                state = 8
        elif state == 8:
            break

        pb.stepSimulation()
        time.sleep(timeStep)
    panda.bullet_client.resetSimulation()
    panda = panda_sim.PandaSim(pb, [0, 0, 0])
    simulate(pb, panda)


if __name__ == '__main__':
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_Y_AXIS_UP, 1)
    pb.setAdditionalSearchPath(pd.getDataPath())

    timeStep = 1. / 60.
    pb.setTimeStep(timeStep)
    pb.setGravity(0, -9.8, 0)
    panda = panda_sim.PandaSim(pb, [0, 0, 0])

    p1 = Process(target=simulate(pb, panda))
    p1.start()


