#!/usr/bin/env python3

import rospy
import numpy as np
from gym import spaces
from drl_navigation import rosbot_env
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
from generate_goal import GenerateRandomGoal
from reward_function import RewardFunction
import os
import math

class RosbotNavigationEnv(rosbot_env.RosbotEnv):
    def __init__(self):
        """
        This Task Env is designed for having the husarion in the husarion world
        closed room with columns.
        It will learn how to move around without crashing.
        """
        # Launch the Task Simulated-Environment
        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/husarion/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="drl_navigation",
                    launch_file_name="training_env.launch",
                    ros_ws_abspath=ros_ws_abspath)

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="drl_navigation",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="environment_params.yaml")

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations
        self.init_linear_forward_speed = rospy.get_param(
            '/husarion/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param(
            '/husarion/init_linear_turn_speed')
        self.max_linear_speed = rospy.get_param('/husarion/max_linear_speed')
        self.max_angular_speed = rospy.get_param('/husarion/max_angular_speed')

        self.new_ranges = rospy.get_param('/husarion/new_ranges')
        self.max_laser_value = rospy.get_param('/husarion/max_laser_value')
        self.min_laser_value = rospy.get_param('/husarion/min_laser_value')

        self.work_space_x_max = rospy.get_param("/husarion/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/husarion/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/husarion/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/husarion/work_space/y_min")

        # Get a random goal
        self.map_yaml_path = rospy.get_param("/husarion/map_yaml_abspath")
        self.map_pgm_path = rospy.get_param("/husarion/map_pgm_abspath")
        self.random_goal = GenerateRandomGoal(self.map_yaml_path, self.map_pgm_path)

        self.start_x, self.start_y, self.start_yaw = (0.0, 0.0, 0.0)

        self.desired_position = Point()
        self.desired_position.x, self.desired_position.y = \
            self.random_goal.generate_random_coordinate(min_distance=0.4, 
                                                        invalid_coordinates=[(self.start_x, self.start_y)])
        self.desired_position.z = 0.0

        self.precision_epsilon = rospy.get_param('/husarion/precision_epsilon')

        self.move_base_precision = rospy.get_param(
            '/husarion/move_base_precision')

        # We create the arrays for the laser readings
        # We also create the arrays for the odometry readings
        # We join them together.

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(RosbotNavigationEnv,
              self).__init__(ros_ws_abspath)

        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        rospy.logdebug("Number of laser readings===>" + str(num_laser_readings))
        high_laser = np.full((num_laser_readings), self.max_laser_value)
        low_laser = np.full((num_laser_readings), self.min_laser_value)

        # We place the Maximum and minimum values of the X,Y and YAW of the odometry
        # The odometry yaw can be any value in the circunference.
        high_pose = np.array([self.work_space_x_max,
                                     self.work_space_y_max,])
        low_pose = np.array([self.work_space_x_min,
                                    self.work_space_y_min])

        self.observation_space = spaces.Dict({
                                    "laser_scan": spaces.Box(low=low_laser,high=high_laser, dtype=np.float32),
                                    "relative_pose": spaces.Box(low=low_pose, high=high_pose, dtype=np.float32),
                                    "heading": spaces.Box(low=-3.14, high=3.14, dtype=np.float32),
                                    "previous_velocity": spaces.Box(low=np.array([0.0, -self.max_angular_speed]), 
                                                                    high=np.array([self.max_linear_speed, self.max_angular_speed]), 
                                                                    dtype=np.float32),
                                })
        
        # Action space
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_angular_speed]),
            high=np.array([self.max_linear_speed, self.max_angular_speed]),
            dtype=np.float32
        )
        
        rospy.logdebug("OBSERVATION SPACE SHAPE===>"+str(self.observation_space.shape))
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" +
                       str(self.observation_space))

        self.reward_function = RewardFunction()

        self.cumulated_steps = 0.0

        self.laser_filtered_pub = rospy.Publisher(
            '/rosbot/laser/scan_filtered', LaserScan, queue_size=1)

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.linear_speed = self.init_linear_forward_speed
        self.angular_speed = self.init_linear_turn_speed
        
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=self.move_base_precision,
                       update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # self.set_initial_pose([self.start_x, self.start_y, self.start_yaw])
        self.reset_amcl_initial_pose([self.start_x, self.start_y, self.start_yaw])

        # For Info Purposes
        self.cumulated_reward = 0.0

        self.index = 0

        self.linear_speed = self.init_linear_forward_speed
        self.angular_speed = self.init_linear_turn_speed

        new_position = Point()
        new_position.x, new_position.y = \
            self.random_goal.generate_random_coordinate(min_distance=0.4, 
                                                        invalid_coordinates=[(self.start_x, self.start_y)])
        self.update_desired_pos(new_position)

        global_pose = self.get_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(
            global_pose.pose.position, self.desired_position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the SumitXl
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))

        self.linear_speed, self.angular_speed = action
        last_action = f" linear_speed:{self.linear_speed}, angular_speed:{self.angular_speed}"

        # We tell Husarion the linear and angular speed to set to execute
        self.move_base(self.linear_speed, self.angular_speed,
                       epsilon=self.move_base_precision, update_rate=10)

        rospy.logdebug("END Set Action ==>"+str(action) +
                       ", ACTION="+str(last_action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        HusarionEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_filtered_scan()

        discretized_laser_scan = self.discretize_scan_observation(laser_scan,
                                                                  self.new_ranges
                                                                  )
        # We get the odometry so that SumitXL knows where it is.
        global_pose = self.get_pose()
        x_position = global_pose.pose.position.x
        y_position = global_pose.pose.position.y
        
        delta_x = round(self.desired_position.x - x_position, 2)
        delta_y = round(self.desired_position.y - y_position, 2)
        yaw = self.get_orientation_euler()[2]

        heading = math.atan2(delta_y, delta_x) - yaw
        normalised_heading = round(self.normalize_angle(heading), 2)

        # We concatenate all the lists.
        observations = {
            "laser_scan": discretized_laser_scan,
            "relative_pose": [delta_x, delta_y],
            "heading": normalised_heading,
            "previous_velocity": [round(self.linear_speed, 2), 
                                  round(self.angular_speed, 2)]
        }

        rospy.logwarn("delta_x: " + str(delta_x))
        rospy.logwarn("delta_y: " + str(delta_y))
        rospy.logwarn("normalised_heading: " + str(normalised_heading))
        rospy.logwarn("linear_speed: " + str(self.linear_speed))
        rospy.logwarn("angular_speed: " + str(self.angular_speed))

        rospy.logwarn("END Get Observation ==>")

        return observations

    def _is_done(self, observations):
        """
        We consider that the episode has finished when:
        1) Husarion has moved ouside the workspace defined.
        2) Husarion is too close to an object
        3) Husarion has reached the desired position
        """

        # We fetch data through the observations
        # Its all the array except from the last four elements, which are XY odom and XY des_pos
        laser_readings = observations["laser_scan"]

        global_pose = self.get_pose()
        current_pos = Point()
        current_pos.x = global_pose.pose.position.x
        current_pos.y = global_pose.pose.position.y
        current_pos.z = 0.0

        rospy.logwarn("is DONE? current_position=" + str(current_pos))
        rospy.logwarn("is DONE? desired_position=" + str(self.desired_position))

        too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        inside_workspace = self.check_inside_workspace(current_pos)
        reached_des_pos = self.check_reached_desired_position(current_pos,
                                                              self.desired_position,
                                                              self.precision_epsilon)

        is_done = too_close_to_object or not(
            inside_workspace) or reached_des_pos

        rospy.logwarn("####################")
        rospy.logwarn("too_close_to_object=" + str(too_close_to_object))
        rospy.logwarn("inside_workspace=" + str(inside_workspace))
        rospy.logwarn("reached_des_pos=" + str(reached_des_pos))
        rospy.logwarn("is_done=" + str(is_done))
        rospy.logwarn("######## END DONE ##")

        return is_done

    def _compute_reward(self, observations, done):

        laser_readings = observations["laser_scan"]

        min_dist_to_obstacle = float('inf')
        for laser_distance in laser_readings:
            if laser_distance < min_dist_to_obstacle:
                min_dist_to_obstacle = laser_distance

        current_pos = Point()
        global_pose = self.get_pose()
        current_pos.x = global_pose.pose.position.x
        current_pos.y = global_pose.pose.position.y
        current_pos.z = 0.0

        relative_pose = observations["relative_pose"]
        distance_from_des_point = np.sqrt(
            relative_pose[0]**2 + relative_pose[1]**2)

        rospy.logwarn("total_distance_from_des_point=" +
                      str(round(self.previous_distance_from_des_point, 2)))
        rospy.logwarn("distance_from_des_point=" +
                      str(round(distance_from_des_point, 2)))

        distance_difference = distance_from_des_point - \
            self.previous_distance_from_des_point
        
        rospy.logwarn("distance_difference=" + str(distance_difference))

        if not done:
            reward = self.reward_function.compute_step_reward(distance_difference=distance_difference,
                                                            distance_to_obstacle=min_dist_to_obstacle,
                                                            linear_speed=self.linear_speed,
                                                            angular_speed=self.angular_speed,
                                                            heading=observations["heading"])
        else:
            reached_des_pos = self.check_reached_desired_position(current_pos,
                                                                  self.desired_position,
                                                                  self.precision_epsilon)

            if reached_des_pos:
                reward = self.reward_function.get_termination_reward(goal_reached=True)
                rospy.logwarn(
                    "GOT TO DESIRED POINT ; DONE, reward=" + str(reward))
            elif self.check_husarion_has_crashed(laser_readings):
                reward = self.reward_function.get_termination_reward(goal_reached=False)
                rospy.logerr(
                    "HUSARION HAS CRASHED ; DONE, reward=" + str(reward))
            else:
                reward = 0
                rospy.logerr(
                    "SOMETHING WENT WRONG ; DONE, reward=" + str(reward))
        
        self.previous_distance_from_des_point = distance_from_des_point
        
        rospy.logwarn("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods
    def update_desired_pos(self, new_position):
        """
        With this method you can change the desired position that you want
        Usarion to be that initialy is set through rosparams loaded through
        a yaml file possibly.
        :new_position: Type Point, because we only value the position.
        """
        self.desired_position.x = new_position.x
        self.desired_position.y = new_position.y

    def discretize_scan_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """

        discretized_ranges = []
        mod = new_ranges

        filtered_range = []

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))
        
        nan_value = (self.min_laser_value + self.min_laser_value) / 2.0

        for i, item in enumerate(data.ranges):
            if (i % mod == 0):
                if item == float('Inf') or np.isinf(item):
                    rospy.logerr("Infinite Value=" + str(item) +
                                 "Assigning Max value")
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    rospy.logerr("Nan Value=" + str(item) +
                                 "Assigning MIN value")
                    discretized_ranges.append(self.min_laser_value)
                else:
                    # We clamp the laser readings
                    if item > self.max_laser_value:
                        discretized_ranges.append(
                            round(self.max_laser_value, 1))
                    elif item < self.min_laser_value:
                        discretized_ranges.append(
                            round(self.min_laser_value, 1))
                    else:
                        # rospy.logwarn(
                        #     "Normal Item, no processing=>" + str(item))
                        discretized_ranges.append(round(item, 1))
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.0)

        self.publish_filtered_laser_scan(laser_original_data=data,
                                         new_filtered_laser_range=filtered_range)

        return discretized_ranges

    def get_orientation_euler(self):
        global_pose = self.get_pose()
        # We convert from quaternions to euler
        orientation_list = [global_pose.pose.orientation.x,
                            global_pose.pose.orientation.y,
                            global_pose.pose.orientation.z,
                            global_pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def get_distance_from_desired_point(self, current_position, desired_position):
        """
        Calculates the distance from the current position to the desired point
        :param current_position:
        :param desired_position:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                desired_position)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def check_husarion_has_crashed(self, laser_readings):
        """
        Based on the laser readings we check if any laser readingdistance is below
        the minimum distance acceptable.
        """
        husarion_has_crashed = False

        for laser_distance in laser_readings:
            # rospy.logwarn("laser_distance==>"+str(laser_distance))
            if laser_distance == self.min_laser_value:
                husarion_has_crashed = True
                rospy.logwarn("HAS CRASHED==>"+str(laser_distance) +
                              ", min="+str(self.min_laser_value))
                break

            elif laser_distance < self.min_laser_value:
                rospy.logerr("Value of laser shouldnt be lower than min==>" +
                             str(laser_distance)+", min="+str(self.min_laser_value))
            elif laser_distance > self.max_laser_value:
                rospy.logerr("Value of laser shouldnt be higher than max==>" +
                             str(laser_distance)+", max="+str(self.min_laser_value))

        return husarion_has_crashed

    def check_inside_workspace(self, current_position):
        """
        We check that the current position is inside the given workspace.
        """
        is_inside = False

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside

    def check_reached_desired_position(self, current_position, desired_position, epsilon=0.1):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = desired_position.x + epsilon
        x_pos_minus = desired_position.x - epsilon
        y_pos_plus = desired_position.y + epsilon
        y_pos_minus = desired_position.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (
            x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (
            y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        rospy.logdebug("###### IS DESIRED POS ? ######")
        rospy.logdebug("epsilon==>"+str(epsilon))
        rospy.logdebug("current_position"+str(current_position))
        rospy.logdebug("x_pos_plus"+str(x_pos_plus) +
                       ",x_pos_minus="+str(x_pos_minus))
        rospy.logdebug("y_pos_plus"+str(y_pos_plus) +
                       ",y_pos_minus="+str(y_pos_minus))
        rospy.logdebug("x_pos_are_close"+str(x_pos_are_close))
        rospy.logdebug("y_pos_are_close"+str(y_pos_are_close))
        rospy.logdebug("is_in_desired_pos"+str(is_in_desired_pos))
        rospy.logdebug("############")

        return is_in_desired_pos

    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):

        length_range = len(laser_original_data.ranges)
        length_intensities = len(laser_original_data.intensities)

        laser_filtered_object = LaserScan()

        h = Header()
        # Note you need to call rospy.init_node() before this will work
        h.stamp = rospy.Time.now()
        h.frame_id = "laser"

        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max

        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            laser_filtered_object.ranges.append(item)
            laser_filtered_object.intensities.append(item)

        self.laser_filtered_pub.publish(laser_filtered_object)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
