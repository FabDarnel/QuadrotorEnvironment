# drones_data_loader.py

import json


def load_drones_data(file_path):
    with open(file_path, 'r') as file:
        drones_data = json.load(file)

    for drone_data in drones_data:
        camera_data = {}

        if drone_data.get('down_facing_camera'):
            camera_data['down_facing'] = {
                'field_of_view_degrees': drone_data.get('down_facing_fov_degrees', 0),
                'resolution_px': drone_data.get('down_facing_resolution_px', [0, 0])
            }
        if drone_data.get('front_facing_camera'):
            camera_data['front_facing'] = {
                'field_of_view_degrees': drone_data.get('front_facing_fov_degrees', 0),
                'resolution_px': drone_data.get('front_facing_resolution_px', [0, 0])
            }
        drone_data['camera_data'] = camera_data

    return drones_data



