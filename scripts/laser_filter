#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(scan_msg, args):
    filtered_ranges = []
    filtered_angles = []
    intensities = []

    min_angle, max_angle = args

    for i, angle in enumerate(scan_msg.angle_min + i * scan_msg.angle_increment 
                              for i in range(len(scan_msg.ranges))):
        if angle < min_angle or angle > max_angle:
            filtered_ranges.append(scan_msg.ranges[i])
            filtered_angles.append(angle)
            intensities.append(scan_msg.intensities[i])

    # Create a new LaserScan message with filtered ranges and angles
    filtered_scan = LaserScan()
    filtered_scan.header = scan_msg.header
    filtered_scan.angle_min = max_angle
    filtered_scan.angle_max = min_angle
    filtered_scan.angle_increment = scan_msg.angle_increment
    filtered_scan.range_min = scan_msg.range_min
    filtered_scan.range_max = scan_msg.range_max
    filtered_scan.ranges = filtered_ranges
    filtered_scan.intensities = intensities

    # Publish the filtered LaserScan message
    filtered_scan_pub.publish(filtered_scan)

if __name__ == '__main__':
    rospy.init_node('laser_filter')

    angle_min = rospy.get_param("/scan_filter/lower_angle")
    angle_max = rospy.get_param("/scan_filter/upper_angle")
    
    scan_sub = rospy.Subscriber("/scan", LaserScan, scan_callback, 
                                callback_args=(angle_min, angle_max))
    filtered_scan_pub = rospy.Publisher('/scan_filtered', LaserScan, queue_size=10)

    rospy.spin()