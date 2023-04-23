# quadrotor_environment.py

# import required libraries
import gym
import numpy as np
import random
from gym import spaces
import pybullet as p
import pybullet_data
import time
from keras.models import load_model
from cnn_algorithm import build_cnn_model
import keras
from functools import partial
from sensor_data import sensor_data
import os
from drone import Drone  # Import the Drone class
from drones_data_loader import load_drones_data

# Define custom environment class inheriting from gym.env
class QuadrotorEnvironment(gym.Env):
    def __init__(self, num_drones, num_obstacles, complexity, selected_drone_data, drones_data):
        self.num_drones = num_drones
        self.num_obstacles = num_obstacles
        self.complexity = complexity
        self.selected_drone_data = selected_drone_data
        self.drones_data = drones_data # Add this line to save the drones_data parameter as an attribute
        self.input_shape = (84, 84, 3) # Modify as needed
        self.num_actions = 10 # Modify as needed
        
        # Initialize the PyBullet simulation environment
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 60)
        self.initialize_drones()
        self.initialize_obstacle_avoidance_algorithm()  # Initialize the obstacle avoidance algorithm
        
        self.model = build_cnn_model(self.input_shape, self.num_actions)

        
        # Load the CNN model
        self.model = load_model("cnn_obstacle_avoidance_model.h5")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # 10 actions

        # Define the observation space:
        # - Position (x, y, z): Each coordinate ranges from 0 to 200.
        # - Velocity (vx, vy, vz): Each velocity component is unbounded, but you can set limits if needed.
        # The shape of the observation space is (num_drones * 6,), where each drone has 6 dimensions (x, y, z, vx, vy, vz).

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.inf, -np.inf, -np.inf] * num_drones),
            high=np.array([200, 200, 200, np.inf, np.inf, np.inf] * num_drones),
            dtype=np.float32)  # Adjust the shape based on your desired observation space
            # Shape: (num_drones * 6,)

        # Initialize environment variables
        self.obstacles = []
        self.drones = []
        self.start_point = self.generate_start_point()  # Define generate_start_point method to create a start point
        self.end_point = self.generate_end_point()  # Define generate_end_point method to create an end point
        self.physics_engine = self.initialize_physics_engine()  # Define initialize_physics_engine method to set up the physics engine
        self.obstacle_avoidance_algorithm = self.initialize_obstacle_avoidance_algorithm()  # Define initialize_obstacle_avoidance_algorithm method to set up the obstacle avoidance algorithm
        self.initialize_obstacle_course()  # Call the initialize_obstacle_course method here

    def initialize_pybullet(self):
        # Code to initialize the PyBullet simulation environment goes here
        pass

    def initialize_obstacle_course(self):
        # Define the size and location of the obstacle course
        obstacle_course_size = [10, 10]
        obstacle_course_position = [0, 0, 0]

        # Add the ground plane to the obstacle course
        self.ground_plane = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\plane.urdf", basePosition=[0, 0, -0.1])

        # Add the walls to the obstacle course
        self.left_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[-5, 0, 1], globalScaling=sum(obstacle_course_size) + 2)
        self.right_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[5, 0, 1], globalScaling=sum(obstacle_course_size) + 2)
        self.front_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[0, 5, 1], globalScaling=2 + sum(obstacle_course_size) + 2)
        self.back_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[0, -5, 1], globalScaling=2 + sum(obstacle_course_size) + 2)

        # Add the obstacles to the obstacle course
        self.obstacle_ids = []
        for i in range(self.num_obstacles):
            obstacle_urdf = self.select_obstacle_urdf()
            obstacle_position = self.generate_obstacle_position(obstacle_course_size, obstacle_course_position)
            obstacle_id = p.loadURDF(obstacle_urdf, basePosition=obstacle_position)
            self.obstacle_ids.append(obstacle_id)

        # Set the camera position and orientation for visualization
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-30,
                                    cameraTargetPosition=[0, 0, 0])

        # Enable physics simulation
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

    def initialize_drones(self):
        # Code to initialize the drones goes here
        pass

    def select_obstacle_urdf(self):
        # This method selects a random URDF file for an obstacle

        urdf_list = ['cube_small.urdf', 'cylinder_small.urdf', 'sphere_small.urdf']

        return random.choice(urdf_list)

    def generate_start_point(self, random_start=False):
        # This method generates the start point for the drone(s) in the environment.
        # If random_start is set to True, the start point will be chosen randomly
        # within the observation space. Otherwise, the start point will be set at (0, 0, 0).
        
        if random_start:
            # Generate a random start point within the observation space
            x = np.random.uniform(self.observation_space.low[0], self.observation_space.high[0])
            y = np.random.uniform(self.observation_space.low[1], self.observation_space.high[1])
            z = np.random.uniform(self.observation_space.low[2], self.observation_space.high[2])
        else:
            # Set the start point at (0, 0, 0)
            x, y, z = 0, 0, 0

        # Return the start point as a numpy array
        return np.array([x, y, z])

    def generate_end_point(self, random_end=True):
        # Generate the end point, ensuring that it's different from the start point
        # and at least 50 x 50 x 50 units away from it.
        while True:
            if random_end:
                # Generate a random end point within the observation space
                x = np.random.uniform(self.observation_space.low[0], self.observation_space.high[0])
                y = np.random.uniform(self.observation_space.low[1], self.observation_space.high[1])
                z = np.random.uniform(self.observation_space.low[2], self.observation_space.high[2])
            else:
                # Set the end point at (200, 200, 200) as a fixed point, if desired
                x, y, z = 200, 200, 200

            end_point = np.array([x, y, z])

            # Calculate the Euclidean distance between the start and end points
            distance = np.linalg.norm(end_point - self.start_point)

            # If the end point is different from the start point and at least 50 x 50 x 50 units away,
            # break the loop and return the end point
            if distance >= 50:
                return end_point

    def initialize_physics_engine(self):
        # Start the PyBullet physics engine
        self.physics_client = p.connect(p.GUI)  # Use p.DIRECT for a non-graphical version

        # Set the gravity for the simulation
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        # Add the PyBullet data path to access built-in models
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Optionally, you can set the time step for the physics simulation.
        # Smaller time steps typically result in more accurate simulations.
        self.time_step = 1.0 / 240.0  # 240 Hz
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)

        # Load the environment, drone, and obstacle models.
        # You need to provide paths to the respective models or use the built-in models.
        self.load_environment()
        self.load_drones()
        self.load_obstacles()

    # Load the environment model
    def load_environment(self):
        # Define the size and location of the environment
        environment_size = [10, 10]
        environment_position = [0, 0, 0]

        # Add the ground plane to the environment
        self.ground_plane = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\plane.urdf", basePosition=[0, 0, -0.1])

        # Add the walls to the environment
        self.left_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[-5, 0, 1], globalScaling=sum(environment_size) + 2)
        self.right_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[5, 0, 1], globalScaling=sum(environment_size) + 2)
        self.front_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[0, 5, 1], globalScaling=2 + sum(environment_size) + 2)
        self.back_wall = p.loadURDF(r"C:\Users\fabri\.conda\envs\new_atm_env\Lib\site-packages\pybullet_data\cube_small.urdf", basePosition=[0, -5, 1], globalScaling=2 + sum(environment_size) + 2)

        # Set the camera position and orientation for visualization
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-30,
                                    cameraTargetPosition=[0, 0, 0])

        # Enable physics simulation
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
    
    # Load the drone model
    def load_drones(self):
        for i in range(self.num_drones):
            # Define the initial position and orientation of the drone
            drone_position = self.start_point
            drone_orientation = p.getQuaternionFromEuler([0, 0, 0])

            # Add the drone to the simulation environment
            drone_urdf = self.select_drone_urdf()
            drone_path = os.path.join(self.drones_dir, drone_urdf)
            drone_id = p.loadURDF(drone_path, drone_position, drone_orientation,
                                physicsClientId=self.physics_engine)
            self.drones.append(Drone(drone_id, drone_position, drone_orientation))  # Use the Drone class

    # This function computes the horizontal velocity, angular velocity, climb rate, descent rate, distance travelled, and time of travel
    def compute_step_metrics(self, drone_pos, prev_pos, elapsed_time):
        # Compute velocity
        velocity = (drone_pos - prev_pos) / elapsed_time

        # Compute climb and descent rate
        climb_rate = max(0, drone_pos[2] - prev_pos[2]) / elapsed_time
        descent_rate = max(0, prev_pos[2] - drone_pos[2]) / elapsed_time

        # Compute distance travelled
        distance = np.linalg.norm(drone_pos - prev_pos)

        # Compute time of travel
        time = elapsed_time

        # Compute angular velocity
        angular_velocity = self.physics_engine.getBaseVelocity(self.drones[0].id)[1]

        return velocity, climb_rate, descent_rate, distance, time, angular_velocity

    # create a dataset by having the drone explore the environment and store the images captured by its camera,
    #  along with the actions taken and the resulting outcomes. You can then use this dataset to train 
    # the CNN model, either online (updating the model as new data becomes available) or 
    # offline (training the model on the collected data after exploration).
    def collect_data(self, drone, action, reward, done, next_state):
            camera_image = drone.get_camera_image()
            self.data.append((camera_image, action, reward, done, next_state))

    # Train the CNN model using the collected data.
    # If you choose online learning, update the model after collecting a certain amount of data (e.g., after every few steps). 
    # This approach can lead to a more adaptive model but may require more computational resources during exploration.
    def train_cnn_online(self):
        if len(self.data) >= self.training_batch_size:
            # Select a batch of data for training
            batch = random.sample(self.data, self.training_batch_size)

            # Preprocess the data and train the model
            X, y = self.preprocess_data(batch)
            self.cnn_model.train_on_batch(X, y)

    # If you choose offline learning, store the data during exploration and train the model afterward. 
    # This approach can be more computationally efficient during exploration but may not adapt as quickly to changes in the environment.
    def train_cnn_offline(self):
        # Preprocess the data and train the model
        X, y = self.preprocess_data(self.data)
        self.cnn_model.fit(X, y, epochs=10, batch_size=32)

    # Function to save the CNN model used for obstacle avoidance
    def save_model(self):
        """
        Saves the trained CNN model to a file named 'cnn_obstacle_avoidance_model.h5'.
        """
        self.cnn_model.save('cnn_obstacle_avoidance_model.h5')

    # Function that takes an input image and outputs the predicted avtion using the loaded model
    def predict_action(self, observation):
        # Reshape the input image to match the expected input shape of the model
        observation = observation.reshape((1, ) + observation.shape)
        # Normalize the image pixels
        observation = observation.astype('float32') / 255.
        # Get the predicted action from the model
        action_probabilities = self.model.predict(observation)
        # Return the action with the highest probability
        return np.argmax(action_probabilities[0])

    # This function takes an action as input, performs it in the environment, collects the data, trains the CNN online or offline based on your choice, 
    # and returns the new observation, reward, done status, and any additional info.
    def step(self, action):
        # You'll need to implement the `perform_action` method
        # that takes an action and updates the drone's position, velocity,
        # and other variables accordingly.
        drone, next_state, reward, done = self.perform_action(action)
        
        # Collect data
        self.collect_data(drone, action, reward, done, next_state)
        
        # Get sensor data from each drone
        sensor_data_results = []
        for drone in self.drones:
            sensor_data_result = drone.get_sensor_data()
            sensor_data_results.append(sensor_data_result)
            
        self.sensor_data_results = sensor_data_results  # save the sensor data results

        # Train the CNN model using the collected data (either online or offline)
        # You can call one of the following methods based on your preference:
        self.train_cnn_online()
        # or
        self.train_cnn_offline()
        
        # Use the CNN model to determine the agent's action
        cnn_action = self.predict_action(self.Observation)
        
        return next_state, reward, done, {}  # modify the return statement if necessary

    # This function provides the elapse time by measuring the time difference between the current and previous simulation steps
    # as commented in the step function
    def perform_action(self, action):
        # Get the current state of the environment
        current_state = self.get_state()

        # Compute the elapsed time since the last simulation step
        if self.last_sim_time is None:
            elapsed_time = 0
        else:
            elapsed_time = time.time() - self.last_sim_time

        # Use the obstacle avoidance algorithm to get the action required to avoid obstacles
        obstacle_avoidance_action = self.obstacle_avoidance_algorithm(current_state)

        # Update the drones' positions and velocities using the selected action and the physics engine
        # Here's an example of how to move the first drone forward by 0.5 units:
        drone = self.drones[0]
        p.setJointMotorControl2(drone.id, 1, p.VELOCITY_CONTROL, targetVelocity=obstacle_avoidance_action, force=drone.max_force)

        # Step the simulation forward by one time step
        p.stepSimulation()

        # Get the new state of the environment
        next_state = self.get_state()

        # Compute the reward, done status, and any additional info based on the new state
        reward = self.compute_reward(next_state)
        done = self.is_done(next_state)
        info = {}

        # Compute the step metrics for each drone
        drone_metrics = {}
        for i, drone in enumerate(self.drones):
            pos = next_state[6*i:6*(i+1)][:3]
            prev_pos = current_state[6*i:6*(i+1)][:3]
            metrics = self.compute_step_metrics(pos, prev_pos, elapsed_time)
            drone_metrics[i] = metrics

            # Compare results with compute_step_metrics function
            function_metrics = self.compute_step_metrics(pos, prev_pos, elapsed_time)
            if function_metrics != metrics:
                print(f"Drone {i}: Metrics don't match! Function: {function_metrics}, Method: {metrics}")

        # Update the last simulation time
        self.last_sim_time = time.time()

        return drone, next_state, reward, done

    # Function to compute energy cost function 
    def energy_function(self, state, action):
        # Calculate the energy consumed based on the drone's weight, battery capacity, and action taken
        weight = self.drones_data['weight_kg']
        battery_capacity = self.drones_data['battery_capacity_Wh']

        # Define a mapping between actions and energy coefficients
        action_energy_coefficients = {
            0: 0.8,
            1: 1.0,
            2: 1.0,
            3: 1.2,
            4: 1.2,
            5: 1.2,
            6: 1.2,
            7: 1.2
        }

        # Use the energy coefficient corresponding to the action taken
        energy_coefficient = action_energy_coefficients[action]

        # Calculate the energy consumed based on the drone's weight, battery capacity, and energy coefficient
        energy = weight * battery_capacity * energy_coefficient

        return energy

    # Function to compute the coverage map of the drone when exploring the unknown environemnt 
    def compute_coverage(self, sensor_data, environment_map):
        """
        Computes the map coverage by comparing the sensor data with the environment map.
        
        Args:
            sensor_data (np.ndarray): A 3D numpy array containing the drone's sensor data. The shape of the array is (height, width, channels).
            environment_map (np.ndarray): A 2D numpy array containing the environment map. The shape of the array is (height, width).
        
        Returns:
            A tuple containing:
            - coverage (float): The map coverage as a percentage, ranging from 0 to 100.
            - cost (float): The cost of the drone's exploration, based on the coverage and other factors.
        """
        
        # Convert the sensor data to grayscale
        sensor_data = np.mean(sensor_data, axis=2)
        
        # Threshold the sensor data to create a binary occupancy map
        occupancy_map = (sensor_data > 0.5).astype(int)
        
        # Compute the number of occupied cells in the occupancy map
        num_occupied_cells = np.sum(occupancy_map)
        
        # Compute the total number of cells in the environment map
        num_total_cells = environment_map.size
        
        # Compute the map coverage as a percentage
        coverage = (num_occupied_cells / num_total_cells) * 100
        
        # Compute the cost based on the coverage and other factors (e.g., energy consumption)
        cost = coverage + 0.1 * np.sum(sensor_data)  # Replace with appropriate cost function
        
        return coverage, cost

    # Function to compute the cost function based on Energy, distance travelled, duration of flight, and map coverage
    def cost_fun(self, state, action):
        # Calculate the energy consumed by the drone based on its action
        energy = self.energy_function(state, action)

        # Calculate the distance to the goal
        distance_to_goal = np.linalg.norm(state[:3] - self.end_point)

        # Calculate the time penalty based on the drone's max speed
        drone_speed = self.drone_data['max_speed_ms']
        max_distance = drone_speed * self.time_step
        time_penalty = 0
        if distance_to_goal > max_distance:
            time_penalty = (distance_to_goal - max_distance) / drone_speed

        # Compute the map coverage and cost
        coverage, map_cost = self.compute_coverage(state[3:], self.environment_map)

        # Combine the energy, distance to goal, time penalty, and map cost to compute the total cost
        total_cost = energy + distance_to_goal + time_penalty + map_cost

        return total_cost

    # This function takes the current state of the environment as input and outputs the predicted action
    # that the drone should take to avoid the obstacles
      
    def initialize_obstacle_avoidance_algorithm(self):
        # Implement logic to initialize the obstacle avoidance algorithm
        # Load the required data
        drone_data = self.drones_data
        drone_id = self.selected_drone_data['drone_id']
        sensor_data_result = sensor_data.sensor_data(self.drones_data, drone_id)
        environment_map = self.environment_map

        print("self.drones_data:", self.drones_data)
        print("sensor_data:", sensor_data)

        # Load the pre-trained obstacle avoidance model
        model_path = 'obstacle_avoidance_model.h5'
        model = keras.models.load_model(model_path)

        # Initialize the algorithm with required data
        self.obstacle_avoidance_algorithm = partial(self.obstacle_avoidance_algorithm, model=model)

        return self.obstacle_avoidance_algorithm


    # Function to check if the environment state is done
    def is_done(self, state):
        # Check if the drone has reached the goal
        distance_to_goal = np.linalg.norm(state[:3] - self.end_point)

        # Check if the drone has crashed into an obstacle
        crashed = False
        for drone in self.drones:
            contact_points = p.getContactPoints(drone.id)
            if len(contact_points) > 0:
                crashed = True
                break

        # Check if the maximum number of simulation steps is reached
        max_steps_reached = self.step_count >= self.max_steps

        # If any of the above conditions is met, the environment is done
        done = (distance_to_goal < self.goal_tolerance) or crashed or max_steps_reached

        return done
    
    # Function to compute the reward based on the current state
    def compute_reward(self, state):
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(state[:3] - self.end_point)

        # Compute the cost function
        cost = self.cost_fun(state, self.last_action_taken)

        # Compute the reward based on the distance to the goal and cost
        reward = -distance_to_goal - cost

        return reward
   