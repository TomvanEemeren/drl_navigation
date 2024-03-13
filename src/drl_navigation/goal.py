import yaml
from PIL import Image
import random
import math

# Load (gmapping) map
def load_map(map_yaml_path, map_pgm_path):
    with open(map_yaml_path, 'r') as yaml_file:
        try:
            map_data = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    map_image = Image.open(map_pgm_path)
    occupancy_data = list(map_image.getdata())

    return map_data, occupancy_data, map_image.width, map_image.height

def generate_random_goal(occupied_coordinates, width, height, resolution, min_distance=0.3):
    while True:
        random_x = random.uniform(0, width * resolution)
        random_y = random.uniform(0, height * resolution)
        
        # Check if the goal is far enough from occupied spaces
        if all(math.sqrt((x - random_x)**2 + (y - random_y)**2) > min_distance for x, y in occupied_coordinates):
            return random_x, random_y

# def get_goal():

#     selected_region = random.choice(regions)
#     x = random.uniform(selected_region['x_min'], selected_region['x_max'])
#     y = random.uniform(selected_region['y_min'], selected_region['y_max'])

#     return x, y

# if __name__ == '__main__':
#     x, y = get_goal()
#     print(f"Random goal: x={x}, y={y}")

    # Office regions
    # regions = [
    #     # Big room
    #     {'name': 'Region 1', 'x_min': -0.8, 'x_max': 0.8, 'y_min': -5, 'y_max': -2.5},
    #     {'name': 'Region 2', 'x_min': 0.8, 'x_max': 3, 'y_min': -3.7, 'y_max': -2.5},
    #     {'name': 'Region 3', 'x_min': 3, 'x_max': 5.2, 'y_min': -5, 'y_max': -4},
    #     # Start room
    #     {'name': 'Region 4', 'x_min': -0.8, 'x_max': 0.5, 'y_min': -1.3, 'y_max': -0.5},
    #     {'name': 'Region 5', 'x_min': 0.5, 'x_max': 1.4, 'y_min': -1.3, 'y_max': 1.3},
    #     # Hallway
    #     {'name': 'Region 4', 'x_min': -0.8, 'x_max': 3.5, 'y_min': 2.5, 'y_max': 3.3},
    #     {'name': 'Region 5', 'x_min': 3, 'x_max': 4, 'y_min': -2.4, 'y_max': 4.1},
    #     # Cabinet room
    #     {'name': 'Region 6', 'x_min': 0.1, 'x_max': -0.8, 'y_min': 4.4, 'y_max': 5.1},
    #     # Empty room
    #     {'name': 'Region 7', 'x_min': 5, 'x_max': 5.8, 'y_min': 2.5, 'y_max': 5.1},
    #     # Box room
    #     {'name': 'Region 8', 'x_min': 5, 'x_max': 5.8, 'y_min': -1.3, 'y_max': 0.2}
    # ]