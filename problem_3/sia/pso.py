import random
from typing import List, Tuple
import numpy as np
import pandas as pd

from utils.matrix.distance_matrix import DistanceMatrix

class PSOParameters:
    """Manages the parameters for the PSO algorithm.

    Attributes:
        n_particles (int): Number of particles initialized.
        c1 (float): Cognitive coefficient that decides the importance of personal best solution.
        c2 (float): Social coefficient that decides the importance of neighborhood best solution.
        inertia_weight (float): Weight applied to the particle's previous velocity, balancing exploration and exploitation.
    """
    def __init__(self, n_particles: int, c1: float = 2.0, c2: float = 2.0, inertia_weight: float = 0.8) -> None:
        self.n_particles = n_particles
        self.c1 = c1
        self.c2 = c2
        self.inertia_weight = inertia_weight

    def update_inertia(self, iteration: int, max_iterations: int, final_weight: float) -> None:
        """Linearly decrease inertia weight over time to reduce exploration."""
        self.inertia_weight = self.inertia_weight - ((self.inertia_weight - final_weight) * (iteration / max_iterations))

class Particle:
    """Represents a particle for solving the PSO.
    
    Attributes:
        routes (List[List[int]]): Routes representing customers visited by each vehicle.
        velocity (List[List[int]]): Placeholder for the velocity of each particle (if using swap moves, etc.)
        best_routes (List[List[int]]): The best routes found by this particle.
        best_fitness (float): Best fitness (shortest distance) achieved by the particle.
    """
    def __init__(self, n_vehicles: int, customer_data: pd.DataFrame, max_capacity: float, distance_matrix: DistanceMatrix, time_windows: List[Tuple[int, int]]):
        self.n_customers = len(customer_data) - 1 # Exclude depot
        self.routes, self.active_vehicles = self._initialize_routes(n_vehicles, customer_data, max_capacity, distance_matrix)
        self.velocity = [[0 for _ in range(self.n_customers)] for _ in range(self.active_vehicles)] # Initialize velocity matching number of customers
        self.best_routes = [route.copy() for route in self.routes]
        self.best_fitness = float('inf')
        self.time_windows = time_windows
    
    def _initialize_routes(self, n_vehicles: int, customer_data: pd.DataFrame, max_capacity: float, distance_matrix: DistanceMatrix) -> Tuple[List[List[int]], int]:
        routes = [[] for _ in range(n_vehicles)]
        unvisited_customers = set(range(1, len(customer_data))) # Exclude depot
        
        current_capacity = [0] * n_vehicles # Track vehicle's capacity
        vehicle_times = [0] * n_vehicles # Track vehicle's time
        active_vehicles = 0
        
        for vehicle in range(n_vehicles):
            routes[vehicle].append(0) # Start at the depot

            if unvisited_customers:
                first_customer = random.choice(list(unvisited_customers))
                routes[vehicle].append(first_customer)
                unvisited_customers.remove(first_customer)

                # Initialize capacity and time based on the starting customer
                current_capacity[vehicle] += customer_data['Demand'].iloc[first_customer]
                vehicle_times[vehicle] = max(vehicle_times[vehicle], customer_data['ReadyTime'].iloc[first_customer]) + customer_data['ServiceTime'].iloc[first_customer]

            while unvisited_customers:
                last_customer = routes[vehicle][-1] # Last customer visited in this route
                nearest_customer = None
                min_distance = float('inf')

                for customer in unvisited_customers:
                    # Calculate the distance to each unvisited customer
                    distance = distance_matrix.matrix[last_customer, customer]
                    customer_demand = customer_data['Demand'].iloc[customer]
                    ready_time = customer_data['ReadyTime'].iloc[customer]
                    due_time = customer_data['Due'].iloc[customer]
                    service_time = customer_data['ServiceTime'].iloc[customer]

                    # Calculate the arrival time considering current route time and distance
                    arrival_time = max(vehicle_times[vehicle] + distance, ready_time)

                    # Check if this customer is feasible
                    if current_capacity[vehicle] + customer_demand <= max_capacity and arrival_time <= due_time:
                        if distance < min_distance:
                            min_distance = distance
                            nearest_customer = customer
                
                if nearest_customer is None:
                    # No feasible customer left for this vehicle, so end the route
                    break

                # Add the nearest feasible customer to the route
                routes[vehicle].append(nearest_customer)
                unvisited_customers.remove(nearest_customer)
                
                # Update vehicle's capacity and time
                current_capacity[vehicle] += customer_data['Demand'].iloc[nearest_customer]
                travel_time = distance_matrix.matrix[last_customer, nearest_customer]
                vehicle_times[vehicle] = max(vehicle_times[vehicle] + travel_time, customer_data['ReadyTime'].iloc[nearest_customer]) + service_time

            # End route by returning to the depot
            routes[vehicle].append(0)
            if len(routes[vehicle]) > 2:  # If the route has customers
                active_vehicles = max(active_vehicles, vehicle + 1)

        routes = routes[:active_vehicles]

        print(f"Initialized routes with {active_vehicles} vehicles: {routes}")
        return routes, active_vehicles

    def evaluate_fitness(self, distance_matrix: DistanceMatrix, locations_df: pd.DataFrame) -> Tuple[float, List[float], List[int]]:
        """Calculates total distance for all routes in a particle"""
        total_distance = 0
        total_penalty = 0
        route_distances = []
        route_violations = []

        for route in self.routes:
            route_distance = 0
            vehicle_time = 0 # Start time for vehicle at depot
            n_route_penalty = 0

            if len(route) > 1:
                # Distance and time from depot to first customer
                route_distance += distance_matrix.matrix[0, route[1]] 
                vehicle_time += distance_matrix.matrix[0, route[1]]

                for i in range(1, len(route) - 1):
                    customer = route[i]
                    next_customer = route[i + 1]

                    arrival_time = vehicle_time
                    ready_time, due_time = self.time_windows[customer]

                    if arrival_time < ready_time:
                        route_distance += 50
                        n_route_penalty += 1
                    elif arrival_time > due_time:
                        route_distance += 100
                        n_route_penalty += 1

                    distance_between_customers = distance_matrix.matrix[customer, next_customer]
                    vehicle_time += distance_between_customers
                    vehicle_time = max(vehicle_time, ready_time)
                    vehicle_time += locations_df['ServiceTime'].iloc[customer]

                    route_distance += distance_between_customers
                
                route_distance += distance_matrix.matrix[route[-2], route[-1]] # Last customer back to depot
                total_penalty += n_route_penalty

            total_distance += route_distance
            route_distances.append(route_distance)
            route_violations.append(n_route_penalty)
            #print(f"[Evaluate Fitness] Calculating fitness for route: {route}, Total route cost: {route_distance}, Total route penalties: {n_route_penalty}")

        print(f"Particle routes: {self.routes}, Fitness: {total_distance}, Number of penalties: {total_penalty}")
        return total_distance, route_distances, route_violations

class PSO:
    """Manages the Particle Swarm Optimization process for solving VRPTW.
    
    Attributes:
        n_particles (int): Number of particles in the swarm.
        n_iterations (int): Number of iterations to run the PSO algorithm.
        particles (List[Particle]): List of particles in the swarm.
        best_global_routes (List[List[int]]): Best routes found across all particles.
        best_global_fitness (float): Best fitness score found across all particles.
    """
    def __init__(self, locations_df: pd.DataFrame, n_vehicles: int, n_iterations: int, pso_parameters: PSOParameters, max_capacity: int):
        self.locations_df = locations_df
        self.n_particles = pso_parameters.n_particles
        self.n_iterations = n_iterations
        self.pso_params = pso_parameters
        self.max_capacity = max_capacity
        self.distance_matrix = DistanceMatrix(locations_df)

        self.demands = locations_df['Demand'].tolist()
        self.time_windows = list(zip(locations_df['ReadyTime'], locations_df['Due']))
        
        # Initialize particles
        self.particles = [Particle(n_vehicles, locations_df, max_capacity, self.distance_matrix, self.time_windows) for _ in range(self.n_particles)]
        max_active_vehicles = max(p.active_vehicles for p in self.particles)

        self.best_global_routes = [[] for _ in range(max_active_vehicles)]
        self.best_global_fitness = float('inf')

    def update_particles(self):
        """Updates the routes of each particle based on customers indices (Not regular velocity)."""
        for particle in self.particles:
            r1 = random.random()
            r2 = random.random()

            # Update route for each vehicle
            for vehicle in range(particle.active_vehicles):
                current_route = particle.routes[vehicle]
                pbest_route = particle.best_routes[vehicle]
                gbest_route = self.best_global_routes[vehicle]
                swap_made = False

                if len(current_route) > 3: # Only swap if route has more than one customer
                    for customer in range(1, len(current_route) - 1):  # Skip depot
                        if swap_made:
                            break

                        # Find the index of the customer in the pbest and gbest routes
                        customer_index_pbest = pbest_route.index(current_route[customer]) if current_route[customer] in pbest_route else None
                        customer_index_gbest = gbest_route.index(current_route[customer]) if current_route[customer] in gbest_route else None

                        # Ensure the customer index exists in both pbest and gbest
                        if customer_index_pbest is not None and customer_index_gbest is not None:
                            # Calculate the positional distance from the current customer to the personal best and global best routess
                            dist_to_pbest = abs(customer_index_pbest - customer)
                            dist_to_gbest = abs(customer_index_gbest - customer)

                            # Calculate swap chance based on distance and randomness
                            swap_chance = (
                                self.pso_params.inertia_weight * particle.velocity[vehicle][customer - 1] +
                                self.pso_params.c1 * r1 * (1 / (dist_to_pbest + 1)) +
                                self.pso_params.c2 * r2 * (1 / (dist_to_gbest + 1))
                            )

                            possible_customers = [i for i in current_route[1:-1] if i != current_route[customer]]
                            if possible_customers:
                                # Swap the position with a randomly selected customer (excluding depot)
                                target_customer = random.choice(possible_customers)

                                # Find the indices of the customers to swap
                                target_customer_index = current_route.index(target_customer)
                                current_customer_index = current_route.index(current_route[customer])

                                if random.random() < swap_chance:
                                    # Ensure the swap is feasible for time windows and capacity constraints
                                    if self.is_swap_feasible(current_route, current_customer_index, target_customer_index):
                                        # Swap the customers in the route
                                        current_route[current_customer_index], current_route[target_customer_index] = current_route[target_customer_index], current_route[current_customer_index]
                                        swap_made = True
                                        
                # Update the particle's routes with the modified route
                particle.routes[vehicle] = current_route.copy()

                # Update best routes if route was changed
                if swap_made:
                    particle.best_routes[vehicle] = current_route.copy()

            fitness, route_distances, route_violations = particle.evaluate_fitness(self.distance_matrix, self.locations_df)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_routes = [route.copy() for route in particle.routes]

    def is_swap_feasible(self, route: List[int], customer_index_1: int, customer_index_2: int) -> bool:
        """Checks if swapping two customers in the same vehicle's route respects the time window and vehicle capacity constraints."""
        # Swap customers
        route[customer_index_1], route[customer_index_2] = route[customer_index_2], route[customer_index_1]
        
        route_capacity = 0
        route_time = 0

        for i in range(1, len(route) - 1):
            customer = route[i]
            prev_customer = route[i-1]

            travel_time = self.distance_matrix.matrix[prev_customer, customer]
            route_time += travel_time
            ready_time, due_time = self.time_windows[customer]
            
            if route_time > due_time:
                return False
            
            # Wait if vehicle show up early
            route_time = max(route_time, ready_time)
            
            route_capacity += self.demands[customer]
            if route_capacity > self.max_capacity:
                return False

        route[customer_index_1], route[customer_index_2] = route[customer_index_2], route[customer_index_1]

        return True

    def optimize(self) -> Tuple[List[List[int]], List[float], List[int], int, List[float], int]:
        best_distances = []
        time_window_violations = []
        max_customers_visited = 0
        total_distances = []

        for i in range(self.n_iterations):
            print(f"\n______________________________________________ITERATION {i + 1}______________________________________________")

            iteration_best_distance = float('inf') # Track best particle distance per iteration

            for j, particle in enumerate(self.particles):
                # Evaluate the fitness of each particle's route
                fitness, route_distances, route_violations = particle.evaluate_fitness(self.distance_matrix, self.locations_df)
                print(f"Particle {j+1} Total cost: {fitness}")
                
                # Update personal bests
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_routes = particle.routes
                
                # Update global best if current particle's fitness is better
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    iteration_best_distance = fitness
                    # Extend or update global best with current particle's best active routes
                    self.best_global_routes = [particle.routes[v][:] if v < particle.active_vehicles else [] for v in range(len(self.best_global_routes))]
                    best_distances = route_distances
                    time_window_violations = route_violations
                    best_solution_index = j
                    max_customers_visited = sum(len(route) - 2 for route in particle.routes)  # Exclude depots

            # Append the best distance for this iteration
            if iteration_best_distance < float('inf'):
                total_distances.append(iteration_best_distance)
            else:
                # Handle the case where no valid distance was found in an iteration (if applicable)
                total_distances.append(self.best_global_fitness)  # or another placeholder

            print(f"Global Best routes: {self.best_global_routes}")
            print(f"Global Best Fitness: {self.best_global_fitness}")
            
            # Update all particle routes based on current bests
            self.update_particles()
            self.pso_params.update_inertia(i, self.n_iterations, final_weight=0.4)
        
        return self.best_global_routes, best_distances, time_window_violations, max_customers_visited, total_distances, best_solution_index

