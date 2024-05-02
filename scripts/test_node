#!/usr/bin/env python3

import math
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def pose_callback(msg):
    # Compute the heading to the x, y coordinate
    target_x = 2.0  # Example x coordinate
    target_y = -2.0  # Example y coordinate

    robot_x = msg.pose.position.x  # Get the robot's x coordinate from the message
    robot_y = msg.pose.position.y  # Get the robot's y coordinate from the message

    # Compute the difference in x and y coordinates
    delta_x = target_x - robot_x
    delta_y = target_y - robot_y

    # Convert pose.orientation to euler angles
    roll, pitch, yaw = euler_from_quaternion([msg.pose.orientation.x, 
                                              msg.pose.orientation.y, 
                                              msg.pose.orientation.z, 
                                              msg.pose.orientation.w])

    # Compute the heading using arctan2
    heading = math.atan2(delta_y, delta_x) - yaw

    heading = abs(normalize_angle(heading))  # Normalize the heading

    # Print the heading
    rospy.loginfo("Heading: " + str(heading))

if __name__ == '__main__':
    rospy.init_node('test_node')  # Initialize the ROS node

    # Create a subscriber to the /scan_filtered topic
    rospy.Subscriber('/rosbot/pose', PoseStamped, pose_callback)

    rospy.spin()  # Spin to keep the node from exiting