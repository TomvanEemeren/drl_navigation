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

def test(env, path_to_model):
    model = SAC.load(path_to_model, env=env)
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action , _  = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1
            if extra_steps <= 0:
                break

def main():
    rospy.init_node('drl_node', anonymous=True, log_level=rospy.INFO)
    
    env_name = rospy.get_param("/husarion/environment_name")

    # # Loads parameters from the ROS param server
    # # Parameters are stored in a yaml file inside the config directory
    # # They are loaded at runtime by the launch file
    model_path = rospy.get_param("/testing/model_abspath")

    gymenv.close()

    if os.path.isfile(model_path):
        gymenv = Start_Environment(env_name)
        rospy.loginfo("Gym environment done")
        test(gymenv, path_to_model=model_path)
    else:
        rospy.logerr(f"Model file not found at {model_path}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass