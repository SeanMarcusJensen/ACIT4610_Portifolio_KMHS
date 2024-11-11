import numpy as np
import pandas as pd
from dataclasses import dataclass

# Utility to calculate Euclidean distance
def calculate_distance(x1, x2, y1, y2) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Data class for customers
@dataclass
class Customer:
    customer_no: int
    lng: int
    lat: int
    demand: int
    ready_time: int
    service_time: int
    
    @staticmethod
    def create_from(df: pd.DataFrame) -> list:
        columns = ['CustomerNO', 'Lng', 'Lat', 'Demand', 'ReadyTime', 'ServiceTime']
        customers = []
        for _, args in df.iterrows():
            customers.append(Customer(*args.loc[columns]))
        return customers

# Vehicle class
class Vehicle:
    def __init__(self, capacity: int, search_space: tuple):
        self.pos = np.random.randint(search_space[0][0], search_space[0][1]), np.random.randint(search_space[1][0], search_space[1][1])
        self.route = []
        self.capacity = capacity
        self.load = 0
        self.velocity = np.array([0.0, 0.0])
        self.best_pos = self.pos

    def __repr__(self):
        return f"Vehicle(pos={self.pos}, route={self.route}, load={self.load})"

# PSO-based Fleet class to manage vehicles
class Fleet:
    def __init__(self, num_vehicles, vehicle_capacity, search_space):
        self.vehicles = [Vehicle(vehicle_capacity, search_space) for _ in range(num_vehicles)]
        self.best_solution = None
        self.best_distance = float('inf')

    def assign_customers(self, customers, distance_map):
        for vehicle in self.vehicles:
            vehicle.route = []
            vehicle.load = 0
        # Assign customers based on priority and feasibility checks
        priority_customers = sorted(customers, key=lambda c: c.ready_time)
        for customer in priority_customers:
            assigned = False
            for vehicle in sorted(self.vehicles, key=lambda v: calculate_distance(customer.lng, *v.pos, customer.lat)):
                if vehicle.load + customer.demand <= vehicle.capacity:
                    insertion_index = find_cheapest_insertion_point(vehicle.route, customer, distance_map)
                    vehicle.route.insert(insertion_index, customer.customer_no)
                    vehicle.load += customer.demand
                    assigned = True
                    break
            if not assigned:
                print(f"Warning: Customer {customer.customer_no} could not be assigned to any vehicle.")

# Initialize distance map
def initialize_distance_map(customers):
    n = len(customers)
    distance_map = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_map[i, j] = calculate_distance(customers[i].lng, customers[j].lng,
                                                    customers[i].lat, customers[j].lat)
    return distance_map

# Find the cheapest insertion point in the vehicle's route
def find_cheapest_insertion_point(route, customer, distance_map):
    if not route:
        return 0  # Start with depot if route is empty
    min_cost = float('inf')
    best_position = 0
    for i in range(len(route) + 1):
        if i == 0:
            cost = distance_map[0, customer.customer_no] + distance_map[customer.customer_no, route[0]]
        elif i == len(route):
            cost = distance_map[route[-1], customer.customer_no] + distance_map[customer.customer_no, 0]
        else:
            cost = (distance_map[route[i-1], customer.customer_no] +
                    distance_map[customer.customer_no, route[i]] -
                    distance_map[route[i-1], route[i]])
        if cost < min_cost:
            min_cost = cost
            best_position = i
    return best_position

# Evaluate the total distance for a fleet
def evaluate_fleet_distance(fleet, distance_map):
    total_distance = 0.0
    for vehicle in fleet.vehicles:
        if vehicle.route:
            for i in range(len(vehicle.route) - 1):
                total_distance += distance_map[vehicle.route[i], vehicle.route[i + 1]]
            total_distance += distance_map[0, vehicle.route[0]]  # Depot to first customer
            total_distance += distance_map[vehicle.route[-1], 0]  # Last customer back to depot
    return total_distance

# PSO algorithm
def particle_swarm_optimization(customers, num_vehicles, vehicle_capacity, num_iterations):
    search_space = ((0, 100), (0, 100))  # Example search space bounds for vehicle positions
    fleet = Fleet(num_vehicles, vehicle_capacity, search_space)
    distance_map = initialize_distance_map(customers)
    global_best_distance = float('inf')
    global_best_position = None

    for iteration in range(num_iterations):
        # Assign customers to vehicles
        fleet.assign_customers(customers, distance_map)
        
        # Evaluate fitness
        total_distance = evaluate_fleet_distance(fleet, distance_map)
        if total_distance < global_best_distance:
            global_best_distance = total_distance
            global_best_position = [vehicle.pos for vehicle in fleet.vehicles]

        # Update positions and velocities
        for vehicle in fleet.vehicles:
            inertia = 0.5  # Inertia weight
            cognitive = 1.5  # Cognitive coefficient
            social = 1.5  # Social coefficient

            # Calculate cognitive and social components
            cognitive_velocity = cognitive * np.random.rand() * (vehicle.best_pos - vehicle.pos)
            social_velocity = social * np.random.rand() * (global_best_position[0] - vehicle.pos)

            # Update velocity and position
            vehicle.velocity = inertia * vehicle.velocity + cognitive_velocity + social_velocity
            vehicle.pos = vehicle.pos + vehicle.velocity
            vehicle.pos = np.clip(vehicle.pos, search_space[0][0], search_space[0][1])

            # Update personal best position
            if evaluate_fleet_distance(fleet, distance_map) < global_best_distance:
                vehicle.best_pos = vehicle.pos

        print(f"Iteration {iteration + 1}/{num_iterations}: Best distance = {global_best_distance}")

    return global_best_distance, global_best_position

# Example setup and running PSO
if __name__ == '__main__':
    # Create sample customers DataFrame
    data = {
        'CustomerNO': range(10),
        'Lng': np.random.randint(0, 100, 10),
        'Lat': np.random.randint(0, 100, 10),
        'Demand': np.random.randint(1, 20, 10),
        'ReadyTime': np.random.randint(0, 50, 10),
        'ServiceTime': np.random.randint(5, 15, 10)
    }
    customers_df = pd.DataFrame(data)
    customers = Customer.create_from(customers_df)

    num_vehicles = 3
    vehicle_capacity = 50
    num_iterations = 100
    best_distance, best_positions = particle_swarm_optimization(customers, num_vehicles, vehicle_capacity, num_iterations)
    print(f"Best distance found: {best_distance}")
    print(f"Best vehicle positions: {best_positions}")

