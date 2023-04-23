# d_star_lite.py

# o implement the D* Lite algorithm, 
# a helper class for the A* search algorithm and the D* Lite algorithm itself is needed
import heapq
import numpy as np

# AStarSearch is a helper class implementing the A* search algorithm
class AStarSearch:
    def __init__(self, cost_fn, heuristic_fn):
        self.cost_fn = cost_fn  # Cost function for the algorithm
        self.heuristic_fn = heuristic_fn  # Heuristic function for the algorithm

    # The search method performs the A* search algorithm
    def search(self, start, goal, neighbors_fn):
        open_set = []  # The set of nodes to be evaluated
        heapq.heappush(open_set, (0, start))  # Push the start node onto the open set
        came_from = {}  # A dictionary of where each node was reached from
        g_score = {start: 0}  # The cost of the cheapest path from the start to each node
        f_score = {start: self.heuristic_fn(start, goal)}  # The total cost of the cheapest path to each node, including the heuristic

        while open_set:  # Continue while there are nodes to be evaluated
            _, current = heapq.heappop(open_set)  # Get the node with the lowest f_score

            if current == goal:  # If the goal is reached, reconstruct and return the path
                return self.reconstruct_path(came_from, current)

            for neighbor in neighbors_fn(current):  # Iterate through the neighbors of the current node
                # Calculate the tentative g_score for the neighbor node
                tentative_g_score = g_score[current] + self.cost_fn(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):  # If the new path is shorter
                    came_from[neighbor] = current  # Update the came_from dictionary
                    g_score[neighbor] = tentative_g_score  # Update the g_score
                    # Update the f_score with the new g_score and the heuristic value
                    f_score[neighbor] = tentative_g_score + self.heuristic_fn(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Push the neighbor onto the open set

        return None  # Return None if no path is found

    # Reconstruct the path from the came_from dictionary
    def reconstruct_path(self, came_from, current):
        path = [current]  # Initialize the path with the current (goal) node
        while current in came_from:  # Continue until the start node is reached
            current = came_from[current]  # Get the previous node in the path
            path.append(current)  # Add the previous node to the path
        return path[::-1]  # Reverse the path before returning