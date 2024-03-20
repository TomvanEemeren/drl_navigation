import rospy    

class RewardFunction:

    def __init__(self):
        self.goal_reached_points = rospy.get_param("/husarion/goal_reached_points")
        self.goal_not_reached_points = rospy.get_param("/husarion/goal_not_reached_points")
        self.c_o = rospy.get_param('/husarion/c_obstacle')
        self.c_d = rospy.get_param('/husarion/c_distance')
        self.c_l = rospy.get_param('/husarion/c_linear')
        self.c_w = rospy.get_param('/husarion/c_angular')
        self.c_h = rospy.get_param('/husarion/c_heading')

        self.max_linear_speed = rospy.get_param('/husarion/max_linear_speed')
        self.w_thershold = rospy.get_param('/husarion/angular_threshold')
        self.d_thershold = rospy.get_param('/husarion/obstacle_threshold')

    def compute_step_reward(self, distance_difference, distance_to_obstacle, 
                            linear_speed, angular_speed, heading):
        """
        Computes the reward for the current state.

        Args:
            distance_to_goal (float): Distance to the goal position.
            distance_to_obstacle (float): Distance to the nearest obstacle.
            linear_speed (float): Current linear speed.
            angular_speed (float): Current angular speed.
            done (bool): Flag indicating if the episode is done.

        Returns:
            float: The computed reward.
        """

        self.distance_difference = distance_difference
        self.distance_to_obstacle = distance_to_obstacle
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.heading = heading

        reward = self.get_distance_reward() + self.get_velocity_reward() \
                    + self.get_rotation_reward() + self.get_obstacle_reward() \
                    + self.get_heading_reward() - 1

        return round(reward, 4)

    def get_termination_reward(self, goal_reached):
        """
        Reward for ending the episode. The reward is negative when
        the episode ends without reaching the goal.

        Returns:
            float: termination reward
        """
        if goal_reached:
            return self.goal_reached_points
        else:
            return self.goal_not_reached_points

    def get_obstacle_reward(self):
        """
        Reward for avoiding obstacles. The reward is negative when
        the distance to an obstacle is within a certain threshold.

        Returns:
            float: obstacle reward
        """
        if self.distance_to_obstacle < self.d_thershold:
            return self.c_o
        else:
            return 0
    
    def get_distance_reward(self):
        """
        Reward for moving closer to the goal

        Returns:
            float: distance reward
        """
        return self.c_d * self.distance_difference
    
    def get_velocity_reward(self):
        """
        Reward for larger forward velocity, punishes slow forward
        velocity exponentionally and discourages backward motion.

        Returns:
            float: velocity reward
        """
        return self.c_l * (self.max_linear_speed - self.linear_speed) ** 2

    def get_rotation_reward(self):
        """
        Reward for low angular velocity, punishes high angular velocity
        to encourage smoother trajectories.

        Returns:
            float: rotational reward
        """
        if abs(self.angular_speed) > self.w_thershold:
            return self.c_w * abs(self.angular_speed)
        else:
            return 0
        
    def get_heading_reward(self):
        return -self.c_h * abs(self.heading)
