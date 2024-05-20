#!/usr/bin/env python3

import os
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from drl_navigation.generate_goal import GenerateRandomGoal

class RandomGoalROSWrapper:
    def __init__(self, start_x, start_y):
        self.start_x = start_x
        self.start_y = start_y

        self.maps_abspath = rospy.get_param("/husarion/maps_abspath", default=None)
        self.min_goal_x = rospy.get_param("/husarion/min_goal_x", default=None)
        self.use_semantics = rospy.get_param("/husarion/use_semantics", default=False)

        self.obstacle_map = rospy.get_param("/costmap/obstacle_map", default=None)
        self.radius = rospy.get_param("/costmap/radius", default=None)

        self.map_yaml_path = self.maps_abspath + self.obstacle_map + ".yaml"
        self.obstacle_map_abspath = self.maps_abspath + self.obstacle_map

        if os.path.exists(self.obstacle_map_abspath + ".png"):
            self.obstacle_map_abspath += ".png"
        elif os.path.exists(self.obstacle_map_abspath + ".pgm"):
            self.obstacle_map_abspath += ".pgm"
        else:
            rospy.logerr("Obstacle map not found.")
            return
                
        self.random_goal = GenerateRandomGoal(self.map_yaml_path, 
                                              self.obstacle_map_abspath, self.radius)
        
        self.goal_pub = rospy.Publisher("/random_goal", PointStamped, queue_size=10)
        if self.use_semantics:
            self.map_pub = rospy.Publisher("/semantic_costmap", Image, queue_size=10)
            self.hazard_pub = rospy.Publisher("/hazard_detected", Bool, queue_size=10)
            self.br = CvBridge()

    def get_costmap(self, x, y, yaw):
        yaw = yaw * 180 / 3.14159265359
        map_image, width, height = self.random_goal.create_costmap(x, y, yaw, 
                                                                   size=(3, 3))
        if map_image is not None:
                self.map_pub.publish(self.br.cv2_to_imgmsg(map_image))
        
        return map_image.astype('uint8'), width, height
    
    def hazard_detected(self, x, y):
        pixel_value = self.random_goal.get_pixel_value(x, y)
        hazard_msg = Bool()
        if pixel_value == 150:
            hazard_msg.data = True
            self.hazard_pub.publish(hazard_msg)
            return True
        hazard_msg.data = True
        self.hazard_pub.publish(hazard_msg)
        return False

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