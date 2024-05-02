#!/usr/bin/env python3

import os
import numpy

# ROS packages required
import rospy
import rospkg

import gym
from stable_baselines3 import SAC

# import our training environment
from drl_navigation.env_utils import Start_Environment

def train(env, timesteps, log_dir, model_dir, model_name, stb3_algo="SAC", batch_size=1024, 
          continue_training=False, model_path=None):
    
    if stb3_algo == "SAC":
        if continue_training:
                model = SAC.load(model_path, env, batch_size=batch_size, tensorboard_log=log_dir)
        else:
            model = SAC("MultiInputPolicy", env, batch_size=batch_size, 
                        verbose=1, device="cuda", tensorboard_log=log_dir)
    else:
        raise ValueError("Invalid stb3_algo value. Supported values are SAC and PPO.")

    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, 
                    tb_log_name=model_name)
        model.save(f"{model_dir}/{model_name}/{stb3_algo}_{timesteps*iters}")

def main():
    rospy.init_node('drl_node', anonymous=True, log_level=rospy.INFO)
    
    env_name = rospy.get_param("/husarion/environment_name")

    gymenv = Start_Environment(env_name)
    rospy.loginfo("Gym environment done")

    training_env = rospy.get_param("/training_env")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('drl_navigation')
    model_dir = pkg_path + "/models/" + training_env
    log_dir = pkg_path + "/logs/" + training_env
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # # Loads parameters from the ROS param server
    # # Parameters are stored in a yaml file inside the config directory
    # # They are loaded at runtime by the launch file
    timesteps = rospy.get_param("/training/timesteps")
    stb3_algo = rospy.get_param("/training/algorithm")
    batch_size = rospy.get_param("/training/batch_size")
    model_name = rospy.get_param("/training/model_name")

    continue_training = rospy.get_param("/training/continue_training", default=False)
    model_path = rospy.get_param("/training/model_path", default=None)

    train(gymenv, timesteps, log_dir, model_dir, model_name,
          stb3_algo, batch_size, continue_training, model_path)

    gymenv.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass