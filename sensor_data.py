# sensor_data.py
from drones_data_loader import load_drones_data

def sensor_data(drones_data, drone_id):
    drone_data = next((d for d in drones_data if d['drone_id'] == drone_id), None)
    if drone_data:
        sensor_data = drone_data['sensor_data']
        # Convert resolution_px to a tuple for easier use later
        sensor_data['resolution_px'] = tuple(sensor_data['resolution_px'])
        return sensor_data
    else:
        return None
