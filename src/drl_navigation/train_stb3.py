#!/usr/bin/env python3

import os
import numpy
import time
import argparse

# ROS packages required
import rospy
import rospkg

import gym
from stable_baselines3 import SAC, PPO

# import our training environment
from drl_navigation.env_utils import Start_Environment

def train(env, timesteps, log_dir, model_dir, stb3_algo="SAC", batch_size=1024):
    if stb3_algo == "SAC":
        model = SAC("MultiInputPolicy", env, batch_size=batch_size, verbose=1, device="cuda", tensorboard_log=log_dir)
    else:
        raise ValueError("Invalid stb3_algo value. Supported values are SAC and PPO.")

    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/{stb3_algo}_{timesteps*iters}")

def main():
    rospy.init_node('drl_node', anonymous=True, log_level=rospy.INFO)
    
    env_name = rospy.get_param("/husarion/environment_name")

    gymenv = Start_Environment(env_name)
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drl_navigation')
    model_dir = pkg_path + "/rl_models"
    log_dir = pkg_path + "/logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # # Loads parameters from the ROS param server
    # # Parameters are stored in a yaml file inside the config directory
    # # They are loaded at runtime by the launch file
    timesteps = rospy.get_param("/training/timesteps")
    stb3_algo = rospy.get_param("/training/algorithm")
    batch_size = rospy.get_param("/training/batch_size")

    start_time = time.time()
    train(gymenv, timesteps, log_dir, model_dir, stb3_algo, batch_size)

    rospy.loginfo("Training time: %s seconds" % (time.time() - start_time))
    gymenv.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass