# obstacle_avoidance.py
# This script provides the obstacle avoidance algorithms required by the utm_gym_env.py 
# to run the initialize_obstacle_avoidance_algorithm function 

from d_star_lite import AStarSearch

# DSLiteObstacleAvoidance is the main class implementing the D* Lite obstacle avoidance algorithm
class DSLiteObstacleAvoidance:
        # Initialize the D* Lite algorithm
        # D* Lite maintains a local map that updates as new information becomes available 
        # (e.g., from the drone's sensors). 
        # The algorithm incrementally updates the A* search tree as the drone encounters 
        # new obstacles and re-plans the path accordingly. This allows the drone to navigate 
        # in unknown environments and adapt to changes in the environment, making it a more 
        # appropriate choice for your use case.
    def __init__(self, cost_fn, heuristic_fn, neighbors_fn, sensor_fn):
        self.a_star = AStarSearch(cost_fn, heuristic_fn)  # Initialize the A* search algorithm
        self.neighbors_fn = neighbors_fn  # Function to get the neighbors of a node
        self.sensor_fn = sensor_fn  # Function to get sensor information from the drone

    # The avoid_obstacles method uses the D* Lite algorithm to avoid obstacles
    def avoid_obstacles(self, drone, environment, start, goal):
        # Get the local map from the drone's sensors
        local_map = self.sensor_fn(drone, environment)
        # Find a path using the A* search algorithm
        path = self.a_star.search(start, goal, lambda pos: self.neighbors_fn(pos, local_map))

