#!/usr/bin/env python3

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

        self.min_goal_x = rospy.get_param("/husarion/min_goal_x", default=None)
        self.max_goal_x = rospy.get_param("/husarion/max_goal_x", default=None)
        self.min_goal_y = rospy.get_param("/husarion/min_goal_y", default=None)
        self.max_goal_y = rospy.get_param("/husarion/max_goal_y", default=None)
        self.use_semantics = rospy.get_param("/husarion/use_semantics", default=False)

        self.map_yaml_abspath = rospy.get_param("/husarion/map_yaml_abspath", default=None)
        self.maps_abspath = rospy.get_param("/husarion/maps_abspath", default=None)
        self.obstacle_maps = rospy.get_param("/costmap/obstacle_maps", default=None)
        self.radius = rospy.get_param("/costmap/radius", default=None)

        self.random_goal = GenerateRandomGoal(self.map_yaml_abspath, self.obstacle_maps, 
                                              self. maps_abspath, self.radius)
        
        self.goal_pub = rospy.Publisher("/random_goal", PointStamped, queue_size=10)
        self.hazard_pub = rospy.Publisher("/hazard_detected", Bool, queue_size=10)
        
        if self.use_semantics:
            self.map_pub = rospy.Publisher("/semantic_costmap", Image, queue_size=10)
            self.br = CvBridge()

    def update_random_map(self):
        self.random_goal.get_random_map()
         
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
                                                        min_x=self.min_goal_x, max_x=self.max_goal_x,
                                                        min_y=self.min_goal_y, max_y=self.max_goal_y)
        
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