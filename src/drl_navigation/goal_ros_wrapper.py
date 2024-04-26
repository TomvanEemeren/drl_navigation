#!/usr/bin/env python3

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from generate_goal import GenerateRandomGoal

class RandomGoalROSWrapper:
    def __init__(self, start_x, start_y):
        self.start_x = start_x
        self.start_y = start_y

        self.map_yaml_path = rospy.get_param("/husarion/map_yaml_abspath")
        self.map_pgm_path = rospy.get_param("/husarion/map_pgm_abspath")
        self.min_goal_x = rospy.get_param("/husarion/min_goal_x", default=None)

        self.random_goal = GenerateRandomGoal(self.map_yaml_path, self.map_pgm_path)
        
        self.goal_pub = rospy.Publisher("/random_goal", PointStamped, queue_size=10)

    def get_random_coordinates(self):
        random_coordinate = \
            self.random_goal.generate_random_coordinate(min_distance=0.4, 
                                                        invalid_coordinates=[(self.start_x, self.start_y)],
                                                        min_x=self.min_goal_x)
        
        # Create Point message
        point_msg = PointStamped()
        point_msg.header = Header()
        point_msg.header.stamp = rospy.Time.now()  # Current time
        point_msg.header.frame_id = "map"
        point_msg.point.x = random_coordinate[0]
        point_msg.point.y = random_coordinate[1]
        point_msg.point.z = 0.0

        # Publish the message
        self.goal_pub.publish(point_msg)

        return random_coordinate