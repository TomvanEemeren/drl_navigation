#!/usr/bin/env python3

from gym.envs.registration import register
from gym import envs
import gym
import rospy

def Start_Environment(task_and_robot_environment_name):
    """
    It Does all the stuff that the user would have to do to make it simpler
    for the user.
    This means:
    0) Registers the TaskEnvironment wanted, if it exists in the Task_Envs.
    2) Checks that the workspace of the user has all that is needed for launching this.
    Which means that it will check that the robot spawn launch is there and the worls spawn is there.
    4) Launches the world launch and the robot spawn.
    5) It will import the Gym Env and Make it.
    """
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    result = Register_Env(task_env=task_and_robot_environment_name,
                                    max_episode_steps=1000)

    if result:
        rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
        env = gym.make(task_and_robot_environment_name)
    else:
        rospy.logwarn("Something Went wrong in the register")
        env = None

    return env

def Register_Env(task_env, max_episode_steps=1000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them
    with variable limits.
    Here is where you have to PLACE YOUR NEW TASK ENV, to be registered and accesible.
    return: False if the Task_Env wasnt registered, True if it was.
    """

    ###########################################################################
    # MovingCube Task-Robot Envs

    result = True
    
    # Husarion Robot
    if task_env == 'RosbotNavigation-v0':

        register(
            id=task_env,
            entry_point='drl_navigation.navigation_env:RosbotNavigationEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from drl_navigation import navigation_env

    # Add here your Task Envs to be registered
    else:
        result = False

    ###########################################################################

    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        #print("REGISTERED GYM ENVS===>"+str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The Task_Robot_ENV given is not Registered ==>" + \
            str(task_env)
        
    return result


def GetAllRegisteredGymEnvs():
    """
    Returns a list of all registered environment IDs in Gym.
    Example: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """
    all_envs = list(envs.registry)
    
    return all_envs