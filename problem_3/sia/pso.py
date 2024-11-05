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
        print(f"Inertia weight updated to: {self.inertia_weight}")

class Particle:
    """Represents a particle for solving the PSO.
    
    Attributes:
        routes (List[List[int]]): Routes representing customers visited by each vehicle.
        velocity (List[List[int]]): Placeholder for the velocity of each particle (if using swap moves, etc.)
        best_position (List[List[int]]): The best routes found by this particle.
        best_fitness (float): Best fitness (shortest distance) achieved by the particle.
    """
    def __init__(self, num_vehicles: int, customer_data: pd.DataFrame, max_capacity: float, distance_matrix: DistanceMatrix):
        self.routes = self._initialize_routes(num_vehicles, customer_data, max_capacity, distance_matrix)
        self.velocity = [[] for _ in range(num_vehicles)] # Initialize velocity for each vehicle's route
        self.best_position = [route.copy() for route in self.routes]
        self.best_fitness = float('inf')
    
    def _initialize_routes(self, num_vehicles: int, customer_data: pd.DataFrame, max_capacity: float, distance_matrix: DistanceMatrix) -> List[List[int]]:
        routes = [[] for _ in range(num_vehicles)]
        
        # Shuffle customers to create diversity in the initial solutions
        customers = list(range(1, len(customer_data))) # Exclude depot
        random.shuffle(customers)
        
        # Distributing customers among the vehicles
        current_capacity = [0] * num_vehicles # To track current capacity for each vehicle
        
        for customer in customers:
            # Find the first vehicle that can accommodate the customer
            for vehicle in range(num_vehicles):
                if current_capacity[vehicle] + customer_data['Demand'].iloc[customer] <= max_capacity:
                    routes[vehicle].append(customer)
                    current_capacity[vehicle] += customer_data['Demand'].iloc[customer]
                    break # Move to the next customer once assigned

        # Ensure routes start and end with the depot (index 0)
        for vehicle in range(num_vehicles):
            routes[vehicle] = [0] + routes[vehicle] + [0]

        print(f"Initialized routes: {routes}")
        return routes

    def evaluate_fitness(self, distance_matrix: DistanceMatrix) -> float:
        """Calculates total distance for all routes in a particle"""
        total_distance = 0

        for route in self.routes:
            route_distance = 0

            if len(route) > 1:
                route_distance += distance_matrix.matrix[0, route[1]] # Depot to first customer

                for i in range(1, len(route) - 1):
                    route_distance += distance_matrix.matrix[route[i], route[i + 1]]
                route_distance += distance_matrix.matrix[route[-2], route[-1]] # Last customer back to depot

            total_distance += route_distance

        print(f"Particle routes: {self.routes}, Fitness: {total_distance}")
        return total_distance

class PSO:
    """Manages the Particle Swarm Optimization process for solving VRPTW.
    
    Attributes:
        n_particles (int): Number of particles in the swarm.
        n_iterations (int): Number of iterations to run the PSO algorithm.
        particles (List[Particle]): List of particles in the swarm.
        best_global_position (List[List[int]]): Best routes found across all particles.
        best_global_fitness (float): Best fitness score found across all particles.
    """
    def __init__(self, locations_df: pd.DataFrame, n_vehicles: int, n_iterations: int, pso_parameters: PSOParameters, max_capacity: int):
        self.n_particles = pso_parameters.n_particles
        self.n_iterations = n_iterations
        self.pso_params = pso_parameters
        self.n_vehicles = n_vehicles
        self.max_capacity = max_capacity
        self.distance_matrix = DistanceMatrix(locations_df)

        self.demands = locations_df['Demand'].tolist()
        self.time_windows = list(zip(locations_df['ReadyTime'], locations_df['Due']))
        
        # Initialize particles
        self.particles = [Particle(n_vehicles, locations_df, max_capacity, self.distance_matrix) for _ in range(self.n_particles)]

        self.best_global_position = [[] for _ in range(n_vehicles)]
        self.best_global_fitness = float('inf')

    def update_particles(self):
        for particle in self.particles:
            r1 = random.random()
            r2 = random.random()

            # Update position for each vehicle
            for vehicle in range(self.n_vehicles):
                current_route = particle.routes[vehicle]
                pbest_route = particle.best_position[vehicle]
                gbest_route = self.best_global_position[vehicle]

                for customer in range(1, len(current_route) -1):
                    # Check if customer index is valid for both personal and global best routes
                    if customer < len(pbest_route) and customer < len(gbest_route):
                        # Calculate the probability of swapping the current customer position
                        swap_chance = (self.pso_params.inertia_weight * (particle.velocity[vehicle][customer] if particle.velocity[vehicle] else 0) +
                                       self.pso_params.c1 * r1 * (pbest_route[customer] != current_route[customer]) +
                                       self.pso_params.c2 * r2 * (gbest_route[customer] != current_route[customer]))
                        
                        if random.random() < swap_chance:
                            # Swap the position of a randomly selected customer with the current customer
                            random_customer = random.randint(1, len(current_route) - 2) # Exclude depot

                            # Avoid swapping the customer with itself
                            if random_customer != customer:
                                current_route[customer], current_route[random_customer] = current_route[random_customer], current_route[customer]
                
                # Update the particle's route with the modified route
                particle.routes[vehicle] = current_route.copy()
                print(f"Updated route for particle (vehicle {vehicle + 1}): {particle.routes[vehicle]}")
            
            fitness = particle.evaluate_fitness(self.distance_matrix)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = [route.copy() for route in particle.routes]

    def optimize(self) -> Tuple[List[List[int]], float]:
        for i in range(self.n_iterations):
            print(f"\nIteration {i + 1}:")

            for particle in self.particles:
                # Evaluate the fitness of each particle's route
                fitness = particle.evaluate_fitness(self.distance_matrix)
                
                # Update personal bests
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.routes
                
                # Update global best if current particle's fitness is better
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = [particle.routes[v][:] for v in range(self.n_vehicles)]

            print(f"Global Best Fitness: {self.best_global_fitness}")
            print(f"Global Best Position: {self.best_global_position}")
            
            # Update all particle velocities and positions based on current bests
            self.update_particles()
            self.pso_params.update_inertia(i, self.n_iterations, final_weight=0.4)
        
        return self.best_global_position, self.best_global_fitness

