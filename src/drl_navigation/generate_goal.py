#!/usr/bin/env python3

import cv2
import yaml
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class GenerateRandomGoal:
    """
    Class to generate random goal coordinates on a map.

    Attributes:
        map_yaml_path (str): Path to the YAML file containing map data.
        map_pgm_path (str): Path to the PGM file containing occupancy data.
        visualise (bool): Flag indicating whether to visualize the map.

    Methods:
        __init__(map_yaml_path, map_pgm_path, visualise=False):
            Initializes the GenerateRandomGoal object.
        load_map(map_yaml_path, map_pgm_path):
            Loads the map data from the YAML and PGM files.
        get_occupied_coordinates(threshold=50):
            Returns a list of occupied coordinates on the map.
        generate_random_goal(min_distance=0.3):
            Generates a random goal coordinate that is far enough from occupied spaces.
        plot_map():
            Plots the map with occupied coordinates.
    """
    def __init__(self, map_yaml_path, map_pgm_path, visualise=False):
        self.map_yaml_path = map_yaml_path
        self.map_pgm_path = map_pgm_path

        self.map_data, self.occupancy_data, self.width, self.height, self.map_image = \
            self.load_map(map_yaml_path, map_pgm_path)
        
        self.resolution = self.map_data['resolution']
        self.origin = self.map_data['origin']
        
        self.occupied_coordinates, self.valid_coordinates = \
            self.get_filtered_coordinates()

        self.contour_info, self.opencv_image = self.get_contours(visualise=visualise)

        if visualise:
            self.plot_map()

    # Load (gmapping) map
    def load_map(self, map_yaml_path, map_pgm_path):
        """
        Loads the map data from the YAML and PGM files.

        Args:
            map_yaml_path (str): Path to the YAML file containing map data.
            map_pgm_path (str): Path to the PGM file containing occupancy data.

        Returns:
            tuple: A tuple containing the map data, occupancy data, width, height, and map image.
        """
        with open(map_yaml_path, 'r') as yaml_file:
            try:
                map_data = yaml.safe_load(yaml_file)
            except yaml.YAMLError as exc:
                print(exc)

        map_image = Image.open(map_pgm_path)
        occupancy_data = list(map_image.getdata())

        return map_data, occupancy_data, map_image.width, map_image.height, map_image
    
    def get_filtered_coordinates(self, lo_threshold=50, hi_threshold=250):
        """
        Returns a list of occupied and valid coordinates on the map. 

        Args:
            lo_threshold (int): Threshold value for occupied coordinates.
            hi_threshold (int): Threshold value for unoccupied coordinates.

        Returns:
            tuple: A tuple containing the occupied and valid coordinates.
        """
        origin_x, origin_y = self.origin[:2]

        occupied_coordinates = []
        valid_coordinates = []

        for row in range(self.height - 1, -1, -1): 
            for col in range(self.width):
                pixel_value = self.occupancy_data[row * self.width + col]
                if isinstance(pixel_value, tuple):
                    rgb_values = pixel_value[:3]
                    if any(value < lo_threshold for value in rgb_values):
                        occupied_coordinates.append((col * self.resolution + origin_x, 
                                                    (self.height - row - 1) * self.resolution + origin_y))
                    elif all(value > hi_threshold for value in rgb_values):
                        valid_coordinates.append((col * self.resolution + origin_x, 
                                                (self.height - row - 1) * self.resolution + origin_y))
                else:
                    if pixel_value < lo_threshold:
                        occupied_coordinates.append((col * self.resolution + origin_x, 
                                                    (self.height - row - 1) * self.resolution + origin_y))
                    elif pixel_value > hi_threshold:
                        valid_coordinates.append((col * self.resolution + origin_x, 
                                                (self.height - row - 1) * self.resolution + origin_y))

        return occupied_coordinates, valid_coordinates
    
    def generate_random_coordinate(self, min_distance=0.3, invalid_coordinates=[], min_x=None, min_y=None):
        """
        Generates a random goal coordinate that is far enough from occupied spaces.

        Args:
            min_distance (float): Minimum distance from occupied spaces.
            invalid_coordinates (list): List of invalid coordinates to avoid.
            min_x (float): Minimum x value for the random coordinate.
            min_y (float): Minimum y value for the random coordinate.

        Returns:
            tuple: A tuple containing the random goal coordinates.
        """
        while True:
            random_coordinate = random.choice(self.valid_coordinates)
            random_x, random_y = random_coordinate[0], random_coordinate[1]
            
            # Check if the goal is far enough from occupied spaces and invalid coordinates
            if all(math.sqrt((x - random_x)**2 + (y - random_y)**2) > min_distance 
                   for x, y in self.occupied_coordinates + invalid_coordinates):
                if min_x is not None and random_x < min_x:
                    continue
                if min_y is not None and random_y < min_y:
                    continue
                return random_x, random_y

    def get_contours(self, visualise=False):
        """
        Returns the contours of red obstacles on the map.

        Args:
            visualise (bool): Flag indicating whether to visualize the map.

        Returns:
            list: A list of contour information.
        """
        opencv_image = np.array(self.map_image.convert('RGB'))
        brg_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([50, 500, 255])

        mask = cv2.inRange(brg_image, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_info = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_info.append({
                'contour': contour,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            })

        if visualise:
            cv2.drawContours(brg_image, contours, -1, (255, 0, 0), 1)
            plt.figure(figsize=(self.width * self.resolution, self.height * self.resolution))
            plt.imshow(brg_image, origin='upper', 
               extent=[self.origin[0], self.origin[0] + self.width * self.resolution, 
                   self.origin[1], self.origin[1] + self.height * self.resolution])
            plt.xlabel('$x$ [m]')  
            plt.ylabel('$y$ [m]') 
            plt.grid(True)
            plt.show()

        return contour_info, brg_image
    
    def draw_bounding_boxes(self, visualise=False):
        for contour in self.contour_info:
            x, y, w, h = contour['x'], contour['y'], contour['width'], contour['height']
            if w <= h:
                points = np.array([[x - h, y], [x + w - 1, y], [x + w - 1, y + h - 1], 
                                   [x - h, y + h - 1]], dtype=np.int32)
                cv2.fillPoly(self.opencv_image, pts=[points], color=(0, 255, 0))
            elif w > h:
                cv2.rectangle(self.opencv_image, (x, y - w), (x + w - 1, y + h - 1), (0, 0, 255), 1)

        if visualise:
            plt.figure(figsize=(self.width * self.resolution, self.height * self.resolution))
            plt.imshow(self.opencv_image, origin='upper', 
               extent=[self.origin[0], self.origin[0] + self.width * self.resolution, 
                   self.origin[1], self.origin[1] + self.height * self.resolution])
            plt.xlabel('$x$ [m]')  
            plt.ylabel('$y$ [m]') 
            plt.grid(True)
            plt.show()

    def plot_map(self, goal_x=None, goal_y=None, start_x=None, start_y=None):
        """
        Plots the map with occupied coordinates.
        """
        plt.figure(figsize=(self.width * self.resolution, self.height * self.resolution))
        plt.imshow(self.map_image, origin='upper', 
               extent=[self.origin[0], self.origin[0] + self.width * self.resolution, 
                   self.origin[1], self.origin[1] + self.height * self.resolution])

        if start_x is not None and start_y is not None:
            plt.plot(start_x, start_y, 'o', color='blue')
            plt.title('Obstacle map with random start')

        if goal_x is not None and goal_y is not None:
            plt.plot(goal_x, goal_y, 'o', color='orange')
            plt.title('Obstacle map with random goal')

        plt.xlabel('$x$ [m]', fontsize=15)  
        plt.ylabel('$y$ [m]', fontsize=15)  
        plt.xticks(fontsize=15)  
        plt.yticks(fontsize=15) 
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    map_yaml_path = "/data/catkin_ws/src/drl_navigation/maps/training_env_one_object.yaml"
    map_pgm_path = "/data/catkin_ws/src/drl_navigation/maps/training_env_one_object.pgm"

    random_goal = GenerateRandomGoal(map_yaml_path, map_pgm_path)
    start_x, start_y = random_goal.generate_random_coordinate(min_distance=0.4)
    goal_x, goal_y = random_goal.generate_random_coordinate(min_distance=0.4, 
                                                            invalid_coordinates=[(start_x, start_y)],
                                                            min_x=None)