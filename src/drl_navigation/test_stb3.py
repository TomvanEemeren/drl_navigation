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

def test(env, path_to_model, num_episodes=100):
    model = SAC.load(path_to_model, env=env)

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

def main():
    rospy.init_node('drl_node', anonymous=True, log_level=rospy.INFO)
    
    gymname = rospy.get_param("/husarion/environment_name")

    # # Loads parameters from the ROS param server
    # # Parameters are stored in a yaml file inside the config directory
    # # They are loaded at runtime by the launch file
    model_path = rospy.get_param("/testing/model_abspath")
    num_episodes = rospy.get_param("/testing/num_episodes")

    if os.path.isfile(model_path):
        gymenv = Start_Environment(gymname)
        rospy.loginfo("Gym environment done")
        test(gymenv, path_to_model=model_path, num_episodes=num_episodes)
        gymenv.close()
    else:
        rospy.logerr(f"Model file not found at {model_path}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass