#!/usr/bin/env python3

import rospy
import tf
from geometry_msgs.msg import PoseStamped

def pose_publisher():
    rospy.init_node('pose_publisher')
    rospy.logwarn("Starting pose_publisher node")

    global_frame = rospy.get_param('/common/global_frame_id')
    robot_base_frame = rospy.get_param('/common/robot_base_frame_id')

    listener = tf.TransformListener()

    pose_pub = rospy.Publisher('rosbot/pose', PoseStamped, queue_size=10)

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform(global_frame, robot_base_frame, rospy.Time(0))

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = global_frame
            pose_msg.pose.position.x = trans[0]
            pose_msg.pose.position.y = trans[1]
            pose_msg.pose.position.z = trans[2]
            pose_msg.pose.orientation.x = rot[0]
            pose_msg.pose.orientation.y = rot[1]
            pose_msg.pose.orientation.z = rot[2]
            pose_msg.pose.orientation.w = rot[3]

            pose_pub.publish(pose_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        rate.sleep()

if __name__ == '__main__':
    try:
        pose_publisher()
    except rospy.ROSInterruptException:
        pass