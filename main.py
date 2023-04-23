# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import load_model
from drones_data_loader import load_drones_data
from quadrotor_environment import QuadrotorEnvironment
import pybullet as p
from sensor_data import sensor_data  # Import the corrected function

# Define action constants
TURN_LEFT = 0
TURN_RIGHT = 1
CLIMB = 2
DESCEND = 3
PITCH = 4
ROLL = 5
YAW = 6
STOP = 7
DECELERATE = 8
ACCELERATE = 9

drones_data = load_drones_data('drones_data.json')
selected_drone_data = drones_data[0]  # Choose a drone from the dataset

num_drones = 1
num_obstacles = 10
complexity = 1

env = QuadrotorEnvironment(num_drones, num_obstacles, complexity, selected_drone_data, drones_data)

env.save_model()
model = load_model("cnn_obstacle_avoidance_model.h5")

# Initialize the obstacle avoidance algorithm with the sensor data for the selected drone
env.initialize_obstacle_avoidance_algorithm(sensor_data(drones_data, 1))  # Use drone_id=1 for now

# Perform actions, collect observations, etc.

num_episodes = 100
episode_rewards = []
episode_lengths = []
# Initialize lists to store metrics
paths = []
coverage = []
distances_travelled = []
time_travelled = []
num_turns = []
max_velocities = []
min_velocities = []
mean_velocities = []
num_collisions = []
num_climbs = []
num_descents = []
computation_times = []
action_counts_list = []

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")

    # Reset the environment to randomize the initial state
    env.reset()

    # Run the simulation
    done = False
    total_reward = 0
    episode_length = 0
    # Initialize episode-specific variables
    path = []
    coverage_percent = 0
    distance_travelled = 0
    time_travelled = 0
    action_counts = {TURN_LEFT: 0, TURN_RIGHT: 0, PITCH: 0, ROLL: 0, YAW: 0, STOP: 0, DECELERATE: 0, ACCELERATE: 0}
    max_velocity = 0
    min_velocity = float('inf')
    mean_velocity = 0
    collisions = 0
    climbs = 0
    descents = 0

    start_time = time.time()
    while not done:
        # Select an action based on the current state
        action = env.obstacle_avoidance_algorithm(env.get_state())

        # Perform the action and get the new state, reward, and done status
        drone, state, reward, done = env.perform_action(action)

        # Update the environment visualization
        env.visualize(state)

        # Print the current reward and state
        print(f"Reward: {reward}, State: {state}")

        # Accumulate rewards and increase episode length
        total_reward += reward
        episode_length += 1

        # Update path, distance travelled, time travelled, turns, and velocities
        path.append(state[:3])
        distance_travelled += np.linalg.norm(state[3:6])
        time_travelled += 1  # Assuming each step is 1 time unit
        max_velocity = max(max_velocity, np.linalg.norm(state[3:6]))
        min_velocity = min(min_velocity, np.linalg.norm(state[3:6]))
        mean_velocity = (mean_velocity * (len(path) - 1) + np.linalg.norm(state[3:6])) / len(path)

        # Update number of collisions
        if len(p.getContactPoints(drone.id)) > 0:
            collisions += 1

        # Update climbs and descents
        if state[2] > state[0]:
            climbs += 1
        elif state[2] < state[0]:
            descents += 1

        # Update turns and other actions
        if action in [TURN_LEFT, TURN_RIGHT, PITCH, ROLL, YAW, STOP, DECELERATE, ACCELERATE]:
            action_counts[action] += 1

    # Perform any necessary calculations, logging, or saving of data after each episode
    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)
    print(f"Episode {episode + 1} completed with total reward {total_reward} and length {episode_length}")

    # Calculate computation time
    computation_time = time.time() - start_time

    # Save episode-specific metrics
    paths.append(path)
    coverage.append(coverage_percent)
    distances_travelled.append(distance_travelled)
    time_travelled.append(time_travelled)
    action_counts_list.append(action_counts)
    max_velocities.append(max_velocity)
    min_velocities.append(min_velocity)
    mean_velocities.append(mean_velocity)
    num_collisions.append(collisions)
    num_climbs.append(climbs)
    num_descents.append(descents)
    computation_times.append(computation_time)

# Calculate and print average reward and episode length
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)
print(f"Average reward over {num_episodes} episodes: {average_reward}")
print(f"Average episode length over {num_episodes} episodes: {average_length}")

# Save results to a file
with open("results.txt", "w") as f:
    f.write(f"Average reward over {num_episodes} episodes: {average_reward}\n")
    f.write(f"Average episode length over {num_episodes} episodes: {average_length}\n")

# Plot and save the simulation paths
plt.figure()
for i, path in enumerate(paths):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], label=f"Episode {i + 1}")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("simulation_paths.png")
plt.show()

action_labels = ['TURN_LEFT', 'TURN_RIGHT', 'PITCH', 'ROLL', 'YAW', 'STOP', 'DECELERATE', 'ACCELERATE']
action_counts_values = list(action_counts.values())

plt.bar(action_labels, action_counts_values)
plt.xlabel('Actions')
plt.ylabel('Count')
plt.title('Action Counts')
plt.show()

action_counts_df = pd.DataFrame(action_counts_list)
print(action_counts_df)

# Print metrics
print("Coverage: ", coverage)
print("Distance travelled: ", distances_travelled)
# ...

# Save metrics in a file
with open("metrics.txt", "w") as f:
    f.write("Coverage: " + str(coverage) + "\n")
    f.write("Distance travelled: " + str(distances_travelled) + "\n")
    # ...



