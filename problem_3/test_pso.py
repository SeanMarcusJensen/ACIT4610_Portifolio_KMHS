import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Dict

class Customer:
    def __init__(self,
                 customer_no: int,
                 lng: int,
                 lat: int,
                 demand: int,
                 ready_time: int,
                 due: int,
                 service_time: int) -> None:
        self.customer_no: int = int(customer_no - 1) # 0-index minus depot
        self.lng: int = int(lng) # x 
        self.lat: int = int(lat) # y
        self.demand: int = int(demand)
        self.ready_time: int = int(ready_time)
        self.due: int = int(due)
        self.service_time: int = int(service_time)

    def copy(self) -> 'Customer':
        return Customer(self.customer_no, self.lng, self.lat, self.demand, self.ready_time, self.due, self.service_time)
    
    @staticmethod
    def create_from(df: pd.DataFrame) -> list['Customer']:
        columns = ['CustomerNO', 'Lng', 'Lat', 'Demand', 'ReadyTime', 'Due', 'ServiceTime']
        customers = []
        for _, args in df.iterrows():
            customers.append(Customer(*args.loc[columns]))
        return customers

class Vehicle:
    CAPACITY = 200
    def __init__(self, w: Tuple, h: Tuple, depot: Customer, dim: int = 2) -> None:
        self.w = w
        self.h = h
        self.position = np.array([np.random.uniform(*w), np.random.uniform(*h)], dtype=float)
        self.velocity = np.zeros(dim, dtype=float)
        self.load = 0
        self.current_time = 0
        self.time_violation = 0
        self.route = []
        self.depot = depot
        self.travel_distance = 0.0

    def copy(self) -> 'Vehicle':
        vehicle = Vehicle(self.w, self.h, self.depot)
        vehicle.position = self.position.copy()
        vehicle.velocity = self.velocity.copy()
        vehicle.load = self.load
        vehicle.current_time = self.current_time
        vehicle.time_violation = self.time_violation
        vehicle.route = [c.copy() for c in self.route]
        vehicle.travel_distance = self.travel_distance
        return vehicle

    def get_fitness(self) -> float:
        route = self.get_route()
        fitness = 0.0
        for i in range(1, len(route)):
            fitness += calculate_distance(route[i].lng, route[i - 1].lng, route[i].lat, route[i - 1].lat)
        self.travel_distance
        return fitness

    def get_route(self) -> List[Customer]:
        return [self.depot.copy(), *[r.copy() for r in self.route], self.depot.copy()]

    def add_customer(self, customer: Customer, index: int) -> None:
        """ Update Load, Current Time etc. """
        self.load += customer.demand

        current_customer = self.route[-1] if len(self.route) > 0 else self.depot

        travel_time = calculate_distance(current_customer.lng, customer.lng, current_customer.lat, customer.lat)
        arrival_time = self.current_time + travel_time
        ready_time, due_time = customer.ready_time, customer.due

        if arrival_time < ready_time:
            self.current_time = ready_time + customer.service_time

        elif arrival_time > due_time:
            self.time_violation += 1
            self.current_time = arrival_time + customer.service_time

        else:
            self.current_time = arrival_time + customer.service_time

        self.route.insert(index, customer)

    def distance_to_customer(self, customer: Customer) -> float:
        return np.sqrt((self.position[0] - customer.lng) ** 2 + (self.position[1] - customer.lat) ** 2)

    def is_customer_feasable(self, customer: Customer) -> bool:
        """ Other constraints...
        Might add time violation, distance violation etc.
        """
        if self.load + customer.demand > self.CAPACITY:
            return False

        time = self.current_time + self.distance_to_customer(customer)
        if time > customer.due:
            return False

        return True

class Particle:
    """ Representation (n + 2m)
    n = number of customers
    2m = number of vehicles * 2 (x, y)
    [n..., x1, y1, x2, y2, ...]
    x1, y1 => Vehicle Prioity
    """
    def __init__(self, depot: Customer, customers: List[Customer], n_vehicles: int, bounds: tuple) -> None:
        self.bounds = bounds
        self.depot = depot
        self.customers = customers
        self.vehicles = np.array([Vehicle(*bounds, depot=depot, dim=2) for _ in range(n_vehicles)], dtype=Vehicle)

        self.pbest_fitness: float = 0.0
        self.pbest_position = [v.copy() for v in self.vehicles]

        self.lbest_fitness: float = 0.0
        self.lbest_position = [v.copy() for v in self.vehicles]

        self.nbest_fitness: float = 0.0
        self.nbest_position = [v.copy() for v in self.vehicles]

    def copy(self) -> 'Particle':
        particle = Particle(self.depot, self.customers, len(self.vehicles), self.bounds)
        particle.vehicles = [v.copy() for v in self.vehicles]
        particle.pbest_position = [v.copy() for v in self.pbest_position]
        particle.lbest_position = [v.copy() for v in self.lbest_position]
        particle.nbest_position = [v.copy() for v in self.nbest_position]
        particle.pbest_fitness = self.pbest_fitness
        particle.lbest_fitness = self.lbest_fitness
        particle.nbest_fitness = self.nbest_fitness
        return particle

    def decode(self):
        # N-Dimension -> Customer Priority List
        customer_priority = self.get_customer_priority_list(self.customers)

        # 2M-Dimension -> Vehicle Priority Matrix
        priority_matrix = self.get_vehicle_priority_matrix(customer_priority)

        # Construct Routes
        self.construct_routes(priority_matrix)

    def get_fitness(self) -> float:
        """ Compare against its PBEST -> ø(Xi) & ø(Pi) """
        fitness = sum([vehicle.get_fitness() for vehicle in self.vehicles])
        if fitness < self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = [v.copy() for v in self.vehicles]
        return fitness

    def update_velocity(self, w: float, c1: float, c2: float, c3: float, gbest_position: List[Vehicle], nbest_position: List[Vehicle]) -> None:
        """
        Update the velocity of each vehicle in the particle with consideration for global best (gbest),
        personal best (pbest), and neighborhood best (nbest).
        
        Parameters:
        - w: Inertia coefficient
        - c1: Cognitive coefficient for personal best (pbest)
        - c2: Social coefficient for global best (gbest)
        - c3: Neighborhood coefficient for neighborhood best (nbest)
        - gbest_position: List of global best positions (for each vehicle)
        - nbest_position: List of neighborhood best positions (for each vehicle)
        """
        max_velocity = 5.0
        for idx, vehicle in enumerate(self.vehicles):
            # Generate three random coefficients for each of the attraction terms
            r1, r2, r3 = np.random.uniform(0, 1, 3)
            
            # Update velocity using pbest, gbest, and nbest
            cognitive = c1 * r1 * (self.pbest_position[idx].position - vehicle.position)
            social = c2 * r2 * (gbest_position[idx].position - vehicle.position)
            neighborhood = c3 * r3 * (nbest_position[idx].position - vehicle.position)
            
            # Calculate the new velocity
            vehicle.velocity = w * vehicle.velocity + cognitive + social + neighborhood
            
            # Apply velocity cap
            speed = np.linalg.norm(vehicle.velocity)
            if speed > max_velocity:
                vehicle.velocity = (vehicle.velocity / speed) * max_velocity

    # def update_velocity(self, w: float, c1: float, c2: float, gbest_position: List[Vehicle]) -> None:
    #     for idx, vehicle in enumerate(self.vehicles):
    #         r1, r2 = np.random.uniform(0, 1, 2)
    #         vehicle.velocity = w * vehicle.velocity + c1 * r1 * (self.pbest_position[idx].position - vehicle.position) + c2 * r2 * (gbest_position[idx].position - vehicle.position)

    def update_position(self) -> None:
        for vehicle in self.vehicles:
            vehicle.position += vehicle.velocity
            vehicle.position = np.clip(vehicle.position, 0, 100)

    @staticmethod
    def get_customer_priority_list(customers: List[Customer]) -> List[Customer]:
        return sorted(customers, key=lambda x: x.ready_time)
    
    def get_vehicle_priority_matrix(self, customers: List[Customer]) -> Dict[Customer, List[Vehicle]]:
        priority: Dict[Customer, List[Vehicle]] = {}
        for customer in customers:
            priority[customer] = self.vehicles[
                    np.argsort([calculate_distance(vehicle.position[0], customer.lng, vehicle.position[1], customer.lat) for vehicle in self.vehicles])].tolist()
        return priority 

    def construct_routes(self, priority_matrix: Dict[Customer, List[Vehicle]]) -> None:
        not_assigned = []
        for customer, vehicles in priority_matrix.items():
            assigned = False
            for vehicle in vehicles:
                if vehicle.is_customer_feasable(customer):
                    best_position = self._find_best_insertion(vehicle.route, customer, DISTANCE_MATRIX)
                    vehicle.add_customer(customer, best_position)
                    assigned = True
                    break

            if not assigned:
                not_assigned.append(customer)

        print(f"Not Assigned: {len(not_assigned)}")

    def _find_best_insertion(self, route: List[Customer], customer: Customer, distance_map: np.ndarray) -> int:
        best_position = 0
        min_cost = float('inf')
        
        for i in range(len(route) + 1):
            cost = 0
            if i > 0:
                cost += distance_map[route[i - 1].customer_no][customer.customer_no]
            if i < len(route):
                cost += distance_map[customer.customer_no][route[i].customer_no]
            if i > 0 and i < len(route):
                cost -= distance_map[route[i - 1].customer_no][route[i].customer_no]
            if cost < min_cost:
                min_cost = cost
                best_position = i
        
        return best_position
    
    def reset(self):
        for vehicle in self.vehicles:
            vehicle.route = []
            vehicle.load = 0
            vehicle.current_time = 0
            vehicle.time_violation = 0
            vehicle.travel_distance = 0.0

    def solution_representation(self):
        customer_ids = [c.customer_no for c in self.customers]
        pos = [(v.position[0], v.position[1]) for v in self.vehicles]
        return [*customer_ids, *pos]

###### HELPERS ######
def cost_function(customers: list[Customer]) -> Callable[[float, float], float]:
    def cost(x: float, y: float) -> float:
        return sum([calculate_distance(c.lng, x, c.lat, y) for c in customers])
    return cost

def calculate_distance(x1, x2, y1, y2) -> float:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_search_space(df: pd.DataFrame) -> tuple:
    min_x, max_x = int(df['Lng'].min()), int(df['Lng'].max())
    min_y, max_y = int(df['Lat'].min()), int(df['Lat'].max())
    return (min_x, max_x), (min_y, max_y)

def get_distance_map(customers: list[Customer]) -> np.ndarray:
    distances = np.zeros((len(customers), len(customers)), dtype=float)
    for i, vi in enumerate(customers):
        for j, vj in enumerate(customers):
            distances[i, j] = calculate_distance(vi.lng, vj.lng, vi.lat, vj.lat)
    return distances

def find_nbest_particle(particle: Particle, neighbors: List[Particle]) -> Tuple[Particle, float]:
    """
    Find the neighbor with the best FDR (Fitness Distance Ratio) for the given particle.
    
    Parameters:
    - particle: The current particle for which to find the neighborhood best (nbest).
    - neighbors: List of neighboring particles to evaluate.
    
    Returns:
    - nbest_particle: The particle with the best FDR.
    - best_fdr: The best FDR value.
    """
    best_fdr = float('inf')
    nbest_particle: Particle = None
    
    for neighbor in neighbors:
        # Calculate the fitness difference and Euclidean distance
        fitness_diff = abs(neighbor.pbest_fitness - particle.get_fitness())
        pos_distance = np.linalg.norm(neighbor.pbest_position[0].position - particle.pbest_position[0].position)
        
        # Calculate FDR, avoid division by zero
        if pos_distance > 0:
            fdr = fitness_diff / pos_distance
            if fdr < best_fdr:
                best_fdr = fdr
                nbest_particle = neighbor

    return nbest_particle, best_fdr

###### HELPERS END ######

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'c101.csv')
    customers = pd.read_csv(DATA_PATH)

    N_ITER = 150
    N_VEHICLES = 14
    N_PARTICLES = 100
    VEHICLE_CAPACITY = 200
    WIDTH, HEIGHT = get_search_space(customers)

    customers = Customer.create_from(customers)
    depot = customers[0]

    DISTANCE_MATRIX = get_distance_map(customers)
    customers = customers[1:]

    K = 3
    W = 0.9
    CG = .5
    CS = 1.5
    CN = 1.5

    gbest_particle: Particle
    gbest_fitness = float('inf')
    gbest_routes = []

    particles = [Particle(depot, customers, N_VEHICLES, (WIDTH, HEIGHT)) for _ in range(N_PARTICLES)]
    for iter in range(N_ITER):
        for particle in particles:
            particle.reset()
            particle.decode()
            fitness = particle.get_fitness()

            """ Update Global Best ø(Xi) & ø(Pg) """
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_particle = particle.copy()
                gbest_routes = [v.get_route() for v in particle.vehicles]

        # Local / Neighbourhood Best
        for particle in particles:
            neigbors = sorted([p for p in particles if p != particle], key=lambda x: calculate_distance(particle.pbest_position[0].position[0], x.pbest_position[0].position[0], particle.pbest_position[0].position[1], x.pbest_position[0].position[1]))[:5]
            best_neigbor = min(neigbors, key=lambda x: x.pbest_fitness)
            particle.lbest_fitness = best_neigbor.pbest_fitness
            particle.lbest_position = [v.copy() for v in best_neigbor.pbest_position]

            nbest_particle, best_fdr = find_nbest_particle(particle, neigbors)
            particle.nbest_fitness = best_fdr
            nbest_position = nbest_particle.pbest_position if nbest_particle else particle.pbest_position
            particle.nbest_position = [v.copy() for v in nbest_position]

        for particle in particles:
            particle.update_velocity(W, CG, CS, CN, gbest_particle.vehicles, particle.nbest_position)
            particle.update_position()

        W = W - ((W - 0.4) * (iter / N_ITER))

        print(f"Iter({iter + 1}/{N_ITER} - fitness: {gbest_fitness}")

    lngs, lats = zip(*[(c.lng, c.lat) for c in customers])
    plt.plot(lngs, lats, 'o', color='red')

    v_lngs, v_lats = zip(*[(p.position[0], p.position[1]) for p in gbest_particle.vehicles])
    plt.plot(v_lngs, v_lats, 'o', color='blue')

    for id, particle in enumerate(particles):
        print(f"Particle {id}: {particle.get_fitness()}")
        for v in particle.vehicles:
            plt.plot(v.position[0], v.position[1], 'x')

    for route in gbest_routes:
        if len(route) > 0:
            lngs, lats = zip(*[(c.lng, c.lat) for c in route])
            plt.plot(lngs, lats, '-o', color='green')

    plt.show()
