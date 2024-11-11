import numpy as np
import pandas as pd
from dataclasses import dataclass

def calculate_distance(x1, x2, y1, y2) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_search_space(df: pd.DataFrame) -> tuple:
    min_x, max_x = int(df['Lng'].min()), int(df['Lng'].max())
    min_y, max_y = int(df['Lat'].min()), int(df['Lat'].max())
    return (min_x, max_x), (min_y, max_y)

class Vehicle:
    def __init__(self, capacity: int, x: tuple, y: tuple):
        """ Initialize Vehicle:
        1. Random Position based on alloweed search space.
            X = U[x_{min}, x_{max}]
            Y = U[y_{min}, y_{max}]
            X, Y \\in I
        2. Initialize Velocity to 0.
        3. Initialize Personal Best to current position.

        POS -> Reference Point.
        """
        self.pos = np.array([np.random.randint(x[0], x[1]), np.random.randint(y[0], y[1])], dtype=int)
        self.vel = np.array([0, 0], dtype=int)
        self.pbest = self.pos.copy() # Make a copy, dont want this to change.
        self.capacity = capacity
        self.load = 0
        self.route = [] # List of CustomerNO.

    def evaluate(self, distance_map: np.ndarray) -> float:
        """ Calculates the travel time from depot through route, and back to depot.
        """
        travel_time = 0.0
        if len(self.route) <= 0:
            return travel_time

        for i in range(0, len(self.route) - 1):
            travel_time += distance_map[self.route[i], self.route[i+1]]

        travel_time += distance_map[self.route[-1], 0] # Back to Depot.

        return travel_time

    def __repr__(self) -> str:
        return f'Vehicle: [pos:{self.pos}, vel:{self.vel}, best: {self.pbest}, load:{self.load}]'

    def __str__(self) -> str:
        return f'Vehicle: [pos:{self.pos}, vel:{self.vel}, best: {self.pbest}, load:{self.load}]'


class Customer:
    def __init__(self,
                 customer_no: int,
                 lng: int,
                 lat: int,
                 demand: int,
                 ready_time: int,
                 service_time: int) -> None:
        self.customer_no: int = int(customer_no - 2) # 0-index minus depot
        self.lng: int = int(lng) # x 
        self.lat: int = int(lat) # y
        self.demand: int = int(demand)
        self.ready_time: int = int(ready_time)
        self.service_time: int = int(service_time)
    
    @staticmethod
    def create_from(df: pd.DataFrame) -> list['Customer']:
        columns = ['CustomerNO', 'Lng', 'Lat', 'Demand', 'ReadyTime', 'ServiceTime']
        customers = []
        for _, args in df.iterrows():
            customers.append(Customer(*args.loc[columns]))
        return customers


class Fleet:
    def __init__(self, n_vehicles: int, vehicle_capacity: int, search_space: tuple):
        self.vehicles = [Vehicle(vehicle_capacity, *search_space) for _ in range(n_vehicles)]
        self.distance_map = None  # Will hold distance between customers after initialization

    def assign_distance_map(self, customers: list[Customer]) -> None:
        n = len(customers)
        self.distance_map = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.distance_map[i, j] = calculate_distance(customers[i].lng, customers[j].lng,
                                                            customers[i].lat, customers[j].lat)


def set_customer_priority_list(customers: list[Customer]) -> list[Customer]:
    # Sort customers by 'ready_time' to establish priority
    return sorted(customers, key=lambda x: x.ready_time)

def set_vehicle_priority_matrix(customers: list[Customer], vehicles: list[Vehicle]):
    priority_matrix = []
    for customer in customers:
        distances = [calculate_distance(customer.lng, vehicle.pos[0], customer.lat, vehicle.pos[1])
                     for vehicle in vehicles]
        # Order vehicles by distance to this customer
        vehicle_order = np.argsort(distances).tolist()
        priority_matrix.append(vehicle_order)

    return priority_matrix

def find_cheapest_insertion_point(route, customer, distance_map):
    if not route:
        # If route is empty, place the customer directly after the depot
        return 0
    
    min_cost = float('inf')
    best_position = 0
    
    # Loop over each possible insertion position in the route
    for i in range(len(route) + 1):
        if i == 0:
            # Insert at the start (from depot to customer, then customer to first customer in route)
            cost = distance_map[0, customer.customer_no] + distance_map[customer.customer_no, route[0]]
        elif i == len(route):
            # Insert at the end (last customer in route to customer, then customer back to depot)
            cost = distance_map[route[-1], customer.customer_no] + distance_map[customer.customer_no, 0]
        else:
            # Insert between two existing customers in the route
            cost = (distance_map[route[i-1], customer.customer_no] +
                    distance_map[customer.customer_no, route[i - 1]] -
                    distance_map[route[i-1], route[i]])
        
        # Update best position if this insertion point has the lowest additional cost
        if cost < min_cost:
            min_cost = cost
            best_position = i
    
    return best_position

def route_construction(customer_priority, vehicle_priority_matrix, fleet, customers):
    print(f"len(customers): {len(customers)}, len(vehicle_priority_matrix): {len(vehicle_priority_matrix)}, len(fleet.vehicles): {len(fleet.vehicles)}")
    for customer in customer_priority:
        assigned = False
        for vehicle_idx in vehicle_priority_matrix[customer.customer_no]:
            vehicle = fleet.vehicles[vehicle_idx]
            if vehicle.load + customer.demand <= vehicle.capacity:
                pos = find_cheapest_insertion_point(vehicle.route, customer, fleet.distance_map)
                vehicle.route.insert(pos, customer.customer_no)
                vehicle.load += customer.demand
                assigned = True
                break
        if not assigned:
            print(f"Customer {customer.customer_no} could not be assigned")


def decode(customers, fleet):
    # 1. Set customer priority list
    customer_priority_list = set_customer_priority_list(customers)
    print(f"Customer Priority List[c: {len(customer_priority_list)}]")

    # 2. Set vehicle priority matrix
    vehicle_priority_matrix = set_vehicle_priority_matrix(customer_priority_list, fleet.vehicles)
    print(f"Vehicle Priority Matrix[x:{len(vehicle_priority_matrix)}]")

    # 3. Construct routes based on priority and constraints
    route_construction(customer_priority_list, vehicle_priority_matrix, fleet, customers)


def plot(customers: list[Customer], vehicles: list[Vehicle]):
    import matplotlib.pyplot as plt
    lngs, lats= zip(*map(lambda x: (x.lng, x.lat) , customers))
    _ = plt.figure(figsize=(20, 10))
    plt.plot(lngs, lats, 'ro', c='red')
    for vehicle in vehicles:
        x, y = vehicle.pos
        plt.plot(x, y, 'go', c='green')
        plt.arrow(x, y, vehicle.vel[0], vehicle.vel[1],
                  color='green', shape='full',
                  length_includes_head=True, width=0.3)
    plt.show()

if __name__ == '__main__':
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'c101.csv')
    customers = pd.read_csv(DATA_PATH)

    N_ITER = 10
    N_VEHICLES = 10
    VEHICLE_CAPACITY = 200
    WIDTH, HEIGHT = get_search_space(customers)

    fleet = Fleet(N_VEHICLES, VEHICLE_CAPACITY, get_search_space(customers))
    vehicles = [Vehicle(VEHICLE_CAPACITY, WIDTH, HEIGHT) for _ in range(N_VEHICLES)]
    customers = Customer.create_from(customers.loc[1:])
    fleet.assign_distance_map(customers)

    plot(customers, vehicles)

    """
    Particle: [n, 2m]
    n = orderer list of customers
    2m = [x1, y1, x2, y2, ..., xm, ym]

    Customers are ordered by priority, but what are priority
    """

    import matplotlib.pyplot as plt
    for i in range(N_ITER):
        decode(customers, fleet)

        # Plot the customers and vehicles
        plot(customers, fleet.vehicles)

        for vehicle in fleet.vehicles:
            print(vehicle.route)
            lngs, lats = zip(*map(lambda x: (x.lng, x.lat), customers))
            plt.plot(lngs, lats,'ro', c='red')
            lats, lngs = zip(*[(customers[customer].lat, customers[customer].lng) for customer in vehicle.route])
            plt.plot(lngs, lats)
        plt.show()
        break

        # 2. Evaluate Route

        # 3. Update Cognitive and Social Best

        # 4. Move Particle


