#!/usr/bin/env python3

import os
import cv2
import yaml
import math
import random
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Map:
    def __init__(self):
        self.occupancy_data = []
        self.width = None
        self.height = None
        self.map_image = None
        self.occupied_coords = []
        self.valid_coords = []
        self.contour_info = []
        self.opencv_image = []
        
    def update_map(self, occupancy_data, width, height, map_image, occupied_coords, 
                   valid_coords, contour_info, opencv_image):
        self.occupancy_data = occupancy_data
        self.width = width
        self.height = height
        self.map_image = map_image
        self.occupied_coords = occupied_coords
        self.valid_coords = valid_coords
        self.contour_info = contour_info
        self.opencv_image = opencv_image

class GenerateRandomGoal:
    """
    Class to generate random goal coordinates on a map.

    Attributes:
        map_yaml_path (str): Path to the YAML file containing map data.
        map_paths (list): List of paths to the PGM files containing occupancy data.
        maps_abspath (str): Absolute path to the maps directory.
        visualise (bool): Flag indicating whether to visualize the map.

    Methods:
        __init__(map_yaml_path, map_paths, maps_abspath, radius=None, visualise=False):
            Initializes the GenerateRandomGoal class.
        load_yaml(map_yaml_path):
            Loads the map data from the YAML file.
        load_map(map_path):
            Loads the map data from the YAML and PGM files.
        get_random_map():
            Returns a random map from the list of maps.
        get_filtered_coordinates(occupancy_data, width, height, lo_threshold=50, hi_threshold=250):
            Returns a list of occupied and valid coordinates on the map.
        generate_random_coordinate(min_distance=0.3, invalid_coordinates=[], min_x=None, min_y=None):
            Generates a random goal coordinate that is far enough from occupied spaces.
        get_contours(map_image, width, height, visualise=False):
            Returns the contours of red obstacles on the map.
        draw_bounding_boxes(opencv_image, contour_info, width, height, visualise=False):
            Draws bounding boxes around the red obstacles on the map.
        create_costmap(x, y, angle, size=(3,3), visualise=False):
            Creates a costmap based on the given parameters.
        translate_image(image, x, y):
            Translates the image so that the specified point (x, y) becomes the center.
        rotate_image(image, angle):
            Rotates the image by a specified angle.
        get_pixel_value(x, y):
            Returns the pixel value at the specified coordinates.
        create_figure(image, resolution, origin, width, height):
            Creates a figure to visualize the map.
        plot_map(map, goal_x=None, goal_y=None, start_x=None, start_y=None):
            Plots the map with occupied coordinates.    
    """

    def __init__(self, map_yaml_path, map_paths, maps_abspath, radius=None, visualise=False):
        self.map_yaml_path = map_yaml_path
        self.map_paths = map_paths
        self.maps_abspath = maps_abspath

        self.map_data = self.load_yaml(map_yaml_path)
        self.resolution = self.map_data['resolution']
        self.origin = self.map_data['origin']
        self.radius = radius / self.resolution if radius is not None else None

        self.map_paths = map_paths
        self.maps = []
        for map_path in self.map_paths:
            occupancy_data, width, height, map_image = self.load_map(map_path)
            occupied_coords, valid_coords = self.get_filtered_coordinates(occupancy_data, width, height)
            contour_info, opencv_image = self.get_contours(map_image, width, height, 
                                                           visualise=visualise)
            self.draw_bounding_boxes(opencv_image, contour_info, width, height, 
                                     visualise=visualise)
            mapdata = Map()
            mapdata.update_map(occupancy_data, width, height, map_image, occupied_coords, 
                               valid_coords, contour_info, opencv_image)
            self.maps.append(mapdata)

        self.current_map = None
        self.get_random_map()

        if visualise:
            self.plot_map(self.current_map)

    def load_yaml(self, map_yaml_path):
        """
        Loads the map data from the YAML file.

        Args:
            map_yaml_path (str): Path to the YAML file containing map data.

        Returns:
            dict: Map data.
        """
        with open(map_yaml_path, 'r') as yaml_file:
            try:
                map_data = yaml.safe_load(yaml_file)
            except yaml.YAMLError as exc:
                print(exc)
        return map_data

    # Load (gmapping) map
    def load_map(self, map_path):
        """
        Loads the map data from the YAML and PGM files.

        Args:
            map_path (str): Path to the PGM or PNG file containing occupancy data.

        Returns:
            tuple: A tuple containing the occupancy data, width, height, and map image.
        """
        map_abspath = self.maps_abspath + map_path
        if os.path.exists(map_abspath + ".png"):
            map_abspath += ".png"
        elif os.path.exists(map_abspath + ".pgm"):
            map_abspath += ".pgm"

        map_image = Image.open(map_abspath)
        occupancy_data = list(map_image.getdata())

        return occupancy_data, map_image.width, map_image.height, map_image
    
    def get_random_map(self):
        self.current_map = random.choice(self.maps)
    
    def get_filtered_coordinates(self, occupancy_data, width, height, lo_threshold=50, hi_threshold=250):
        """
        Returns a list of occupied and valid coordinates on the map. 

        Args:
            occupancy_data (list): List of occupancy data.
            width (int): Width of the map.
            height (int): Height of the map.
            lo_threshold (int): Threshold value for occupied coordinates.
            hi_threshold (int): Threshold value for unoccupied coordinates.

        Returns:
            tuple: A tuple containing the occupied and valid coordinates.
        """
        origin_x, origin_y = self.origin[:2]

        occupied_coordinates = []
        valid_coordinates = []

        for row in range(height - 1, -1, -1): 
            for col in range(width):
                pixel_value = occupancy_data[row * width + col]
                if isinstance(pixel_value, tuple):
                    rgb_values = pixel_value[:3]
                    if any(value < lo_threshold for value in rgb_values):
                        occupied_coordinates.append((col * self.resolution + origin_x, 
                                                    (height - row - 1) * self.resolution + origin_y))
                    elif all(value > hi_threshold for value in rgb_values):
                        valid_coordinates.append((col * self.resolution + origin_x, 
                                                (height - row - 1) * self.resolution + origin_y))
                else:
                    if pixel_value < lo_threshold:
                        occupied_coordinates.append((col * self.resolution + origin_x, 
                                                    (height - row - 1) * self.resolution + origin_y))
                    elif pixel_value > hi_threshold:
                        valid_coordinates.append((col * self.resolution + origin_x, 
                                                (height - row - 1) * self.resolution + origin_y))

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
        valid_coordinates = self.current_map.valid_coords
        occupied_coordinates = self.current_map.occupied_coords
        while True:
            random_coordinate = random.choice(valid_coordinates)
            random_x, random_y = random_coordinate[0], random_coordinate[1]
            
            # Check if the goal is far enough from occupied spaces and invalid coordinates
            if all(math.sqrt((x - random_x)**2 + (y - random_y)**2) > min_distance 
                   for x, y in occupied_coordinates + invalid_coordinates):
                if min_x is not None and random_x < min_x:
                    continue
                if min_y is not None and random_y < min_y:
                    continue
                return random_x, random_y

    def get_contours(self, map_image, width, height, visualise=False):
        """
        Returns the contours of red obstacles on the map.

        Args:
            map_image (PIL.Image): The map image.
            width (int): Width of the map.
            height (int): Height of the map.
            visualise (bool): Flag indicating whether to visualize the map.

        Returns:
            list: A list of contour information.
        """
        opencv_image = np.array(map_image.convert('RGB'))
        bgr_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        
        lower_red = np.array([0, 0, 100])
        upper_red = np.array([50, 500, 255])

        mask = cv2.inRange(bgr_image, lower_red, upper_red)
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

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) 

        if visualise:
            cv2.drawContours(gray_image, contours, -1, 0, 1)
            self.create_figure(gray_image, self.resolution, 
                               self.origin, width, height)
            plt.xlabel('$x$ [m]')  
            plt.ylabel('$y$ [m]') 
            plt.show()

        return contour_info, gray_image
    
    def draw_bounding_boxes(self, opencv_image, contour_info, width, height, visualise=False):
        """
        Draws bounding boxes around the red obstacles on the map.

        Args:
            opencv_image (np.array): The map image in OpenCV format.
            contour_info (list): List of contour information.
            width (int): Width of the map.
            height (int): Height of the map.
            visualise (bool): Flag indicating whether to visualize the map.
        """
        for contour in contour_info:
            x, y, w, h = contour['x'], contour['y'], contour['width'], contour['height']
            if self.radius is not None:
                if w <= h:
                    # Draw left box
                    points = np.array([[x - self.radius, y], [x - 1, y], [x - 1, y + h - 1], 
                                    [x - self.radius, y + h - 1]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                    # Draw right box
                    points = np.array([[x + w + self.radius, y], [x + w, y], [x + w, y + h - 1], 
                                    [x + w + self.radius, y + h - 1]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                elif w > h:
                    # Draw top box
                    points = np.array([[x, y - self.radius], [x, y - 1], [x + w - 1, y - 1],
                                    [x + w - 1, y - self.radius]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                    # Draw bottom box
                    points = np.array([[x, y + h + self.radius], [x, y + h], [x + w - 1, y + h],
                                    [x + w - 1, y + h + self.radius]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
            else:
                if w <= h:
                    # Draw left box
                    points = np.array([[x - h, y], [x - 1, y], [x - 1, y + h - 1], 
                                    [x - h, y + h - 1]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                    # Draw right box
                    points = np.array([[x + w + h, y], [x + w, y], [x + w, y + h - 1], 
                                    [x + w + h, y + h - 1]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                elif w > h:
                    # Draw top box
                    points = np.array([[x, y - w], [x, y - 1], [x + w - 1, y - 1],
                                    [x + w - 1, y - w]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)
                    # Draw bottom box
                    points = np.array([[x, y + h + w], [x, y + h], [x + w - 1, y + h],
                                    [x + w - 1, y + h + w]], dtype=np.int32)
                    cv2.fillPoly(opencv_image, pts=[points], color=150)

        if visualise:
            self.create_figure(opencv_image, self.resolution, 
                               self.origin, width, height)
            plt.xlabel('$x$ [m]')  
            plt.ylabel('$y$ [m]') 
            plt.show()

    def create_costmap(self, x, y, angle, size=(3,3), visualise=False):
        """
        Creates a costmap based on the given parameters.

        Args:
            x (float): The x-coordinate of the center of the costmap.
            y (float): The y-coordinate of the center of the costmap.
            angle (float): The angle (in radians) to rotate the costmap.
            size (tuple, optional): The size of the costmap in meters (width, height). Defaults to (3, 3).
            visualise (bool, optional): Whether to visualize the costmap. Defaults to False.

        Returns:
            numpy.ndarray: The local costmap image.

        """
        height = self.current_map.height
        opencv_image = self.current_map.opencv_image

        pixel_x = int((x - self.origin[0]) / self.resolution)
        pixel_y = height - int((y - self.origin[1]) / self.resolution) - 1
        size_x = int(size[0] / self.resolution)
        size_y = int(size[1] / self.resolution)

        if visualise:
            cv2.circle(opencv_image, (pixel_x, pixel_y), radius=1, color=0, thickness=-1)
            self.create_figure(opencv_image, self.resolution,
                self.origin, opencv_image.shape[1], opencv_image.shape[0])
            plt.show()

        translated_image, new_center = self.translate_image(opencv_image, pixel_x, pixel_y)
        
        min_x = max(0, new_center[0] - size_x // 2)
        max_x = min(translated_image.shape[0], (new_center[0] + size_x // 2) + 1)
        min_y = max(0, new_center[1] - size_y // 2)
        max_y = min(translated_image.shape[1], (new_center[1] + size_y // 2) + 1)

        rotated_image = self.rotate_image(translated_image, angle)
        cropped_image = rotated_image[int(min_y):int(max_y), int(min_x):int(max_x)]

        if visualise:
            self.create_figure(cropped_image, self.resolution,
                self.origin, cropped_image.shape[1], cropped_image.shape[0])
            plt.show()

        width = cropped_image.shape[1]
        height = cropped_image.shape[0]

        return cropped_image, width, height
    
    def translate_image(self, image, x, y):
        """
        Translates the image so that the specified point (x, y) becomes the center.

        Args:
            image (np.array): Image to be translated.
            x (int): x-coordinate of the center point.
            y (int): y-coordinate of the center point.

        Returns:
            np.array: Translated image.
        """
        center = tuple(np.array(image.shape[1::-1]) / 2)
        tx = -(x - center[0])
        ty = -(y - center[1])
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=220)
        new_center = tuple(np.array(translated_image.shape[1::-1]) / 2)
        return translated_image, new_center
    
    def rotate_image(self, image, angle):
        """
        Rotates the image by a specified angle.

        Args:
            image (np.array): Image to be rotated.
            angle (float): Angle of rotation.

        Returns:
            np.array: Rotated image.
        """
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=220)
        return rotated_image
    
    def get_pixel_value(self, x, y):
        """
        Returns the pixel value at the specified coordinates.

        Args:
            x (int): x-coordinate of the pixel.
            y (int): y-coordinate of the pixel.

        Returns:
            int: Pixel value.
        """
        height = self.current_map.height
        width = self.current_map.width

        pixel_x = int((x - self.origin[0]) / self.resolution)
        pixel_y = height - int((y - self.origin[1]) / self.resolution) - 1
        
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            return None
        
        return self.current_map.opencv_image[pixel_y, pixel_x]

    def create_figure(self, image, resolution, origin, width, height):
        plt.figure(figsize=(width * resolution, height * resolution))
        plt.imshow(image, origin='upper', cmap="gray",
            extent=[origin[0], origin[0] + width * resolution, 
                origin[1], origin[1] + height * resolution])
        plt.grid(True)

    def plot_map(self, map, goal_x=None, goal_y=None, start_x=None, start_y=None):
        """
        Plots the map with occupied coordinates.
        """
        self.create_figure(map.map_image, self.resolution, 
                           self.origin, map.width, map.height)

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
        plt.show()

if __name__ == '__main__':
    map_yaml_path = "/data/catkin_ws/src/drl_navigation/maps/revit_one_object/revit_one_object.yaml"
    map_pgm_path = ["revit_one_object_2","revit_one_object_3","revit_one_object_4","revit_one_object_5"]
    maps_abspath = "/data/catkin_ws/src/drl_navigation/maps/revit_one_object/"
    random_goal = GenerateRandomGoal(map_yaml_path, map_pgm_path, maps_abspath, radius=1.2, visualise=False)
    start_x, start_y = random_goal.generate_random_coordinate(min_distance=0.4)
    goal_x, goal_y = random_goal.generate_random_coordinate(min_distance=0.4, 
                                                            invalid_coordinates=[(start_x, start_y)],
                                                            min_x=None)
    start_time = time.time()
    image, width, height = random_goal.create_costmap(0, 0, 0, size=(3, 3), visualise=True)
    difference = time.time() - start_time
    pixel_value = random_goal.get_pixel_value(3.22, 0.6)
    print(f"Width: {width}, Height: {height}")
    print(f"Pixel value: {pixel_value}")
    print(f"Time taken: {difference:.4f} seconds")