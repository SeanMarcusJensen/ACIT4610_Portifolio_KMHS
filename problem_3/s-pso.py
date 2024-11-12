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
        self.customer_no: int = int(customer_no) # 0-index minus depot
        self.lng: int = int(lng) # x 
        self.lat: int = int(lat) # y
        self.demand: int = int(demand)
        self.ready_time: int = int(ready_time)
        self.due: int = int(due)
        self.service_time: int = int(service_time)
    
    @staticmethod
    def create_from(df: pd.DataFrame) -> list['Customer']:
        columns = ['CustomerNO', 'Lng', 'Lat', 'Demand', 'ReadyTime', 'Due', 'ServiceTime']
        customers = []
        for _, args in df.iterrows():
            customers.append(Customer(*args.loc[columns]))
        return customers

class Vehicle:
    CAPACITY = 200
    INERTIA = 0.9
    COGNITIVE = 1.2
    SOCIAL = 0.5
    LOCAL = 1.5
    NEIGHBOR = 1.5

    def __init__(self, id: int, depot: Customer, width: tuple, height: tuple) -> None:
        self.id = id
        self.bounds = (width, height)
        self.pos = np.array([np.random.uniform(*width), np.random.uniform(*height)], dtype=float)
        self.velocity = np.array(np.random.uniform((-1, 1), 2), dtype=float)
        self.velocity_cap = ((-0.2 * (width[1] - width[0]), 0.2 * (width[1] - width[0])), (-0.2 * (height[1] - height[0]), 0.2 * (height[1] - height[0])))
        self.route: List[Customer] = []
        self.pbest = self.pos.copy()
        self.load = 0.0
        self.current_time = 0.0
        self.depot = depot
        self.due_timer = depot.due
        self.time_violation = 0

    def copy(self) -> 'Vehicle':
        particle = Vehicle(self.id, self.depot, self.bounds[0], self.bounds[1])
        particle.pos = self.pos.copy()
        particle.velocity = self.velocity.copy()
        particle.pbest = self.pbest.copy()
        particle.route = self.route.copy()
        particle.load = self.load
        particle.current_time = self.current_time
        particle.time_violation = self.time_violation
        return particle

    def add_customer(self, customer: Customer) -> None:
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
        self.route.append(customer)

    def order_route(self, distance_map: List[List[float]]) -> List[Customer]:
        def is_2opt_beneficial(i: int, j: int, distance_map: List[List[float]]) -> bool:
            current_dist = (distance_map[self.route[i - 1].customer_no - 1][self.route[i].customer_no - 1] +
                            distance_map[self.route[j - 1].customer_no - 1][self.route[j].customer_no - 1])
            new_dist = (distance_map[self.route[i - 1].customer_no - 1][self.route[j - 1].customer_no - 1] +
                        distance_map[self.route[i].customer_no - 1][self.route[j].customer_no - 1])
            return new_dist < current_dist

        self.route = sorted(self.route, key=lambda x: x.ready_time)

        improved = True
        while improved:
            improved = False
            for i in range(1, len(self.route) - 2):  # Skip depot at start and end
                for j in range(i + 1, len(self.route)):
                    if is_2opt_beneficial(i, j, distance_map):
                        self.route[i:j] = reversed(self.route[i:j])
                        improved = True
        
        return [depot] + self.route + [depot]  # Return route with depot at start and end

    def distance_to_customer(self, customer: Customer) -> float:
        return np.sqrt((self.pos[0] - customer.lng) ** 2 + (self.pos[1] - customer.lat) ** 2)

    def is_customer_feasable(self, customer: Customer) -> bool:
        """ Other constraints...
        Might add time violation, distance violation etc.
        """
        if self.load + customer.demand > self.CAPACITY:
            return False

        # If not in time, return False
        # TODO: MIGHT WANT TO EASE THIS.
        time = self.current_time + self.distance_to_customer(customer)
        if time > customer.due:
            return False

        return True

    def evaluate(self, cost_function: Callable[[float, float], float]) -> float:
        return cost_function(*self.pos)

    def evaluate_pbest(self, cost_function: Callable[[float, float], float]) -> float:
        return cost_function(*self.pbest)

    def update_velocity(self, gbest: 'Vehicle', lbest: 'Vehicle', nbest: 'Vehicle') -> None:
        R1, R2, R3, R4 = np.random.rand(4)
        cog = self.COGNITIVE * R1 * (self.pbest - self.pos)
        glob = self.SOCIAL * R2 * (gbest.pos - self.pos)
        local = self.LOCAL * R3 * (lbest.pos - self.pos)
        neigh = self.NEIGHBOR * R4 * (nbest.pos - self.pos)
        self.velocity = self.INERTIA * self.velocity + cog + glob + local + neigh 
        self.velocity[0] = np.clip(self.velocity[0], *self.velocity_cap[0])
        self.velocity[1] = np.clip(self.velocity[1], *self.velocity_cap[1])

    def update_position(self) -> None:
        self.pos += self.velocity
        self.pos[0] = np.clip(self.pos[0], *self.bounds[0])
        self.pos[1] = np.clip(self.pos[1], *self.bounds[1])

    def __str__(self) -> str:
        return f'Particle[id: {self.id}]'

    def __repr__(self) -> str:
        return f'Particle[id: {self.id}]'

class Particle:
    def __init__(self, n_vehicles: int, bounds: tuple, depot: Customer) -> None:
        self.particles = np.array([Vehicle(id+1, depot, *bounds) for id in range(n_vehicles)], dtype=Vehicle)
        self.pbest = self.particles.copy()

    @staticmethod
    def get_customer_priority_list(customers: List[Customer]) -> List[Customer]:
        return sorted(customers, key=lambda x: x.ready_time)
    
    def get_vehicle_priority_matrix(self, customers: List[Customer]) -> Dict[Customer, List[Vehicle]]:
        priority: Dict[Customer, List[Vehicle]] = {}
        for customer in customers:
            priority[customer] = self.particles[
                    np.argsort([vehicle.distance_to_customer(customer) for vehicle in self.particles])] \
                            .tolist()
        return priority 
    
    def construct_routes(self, priority_matrix: Dict[Customer, List[Vehicle]]) -> None:
        not_assigned = []
        for customer, vehicles in priority_matrix.items():
            assigned = False
            for vehicle in vehicles:
                if vehicle.is_customer_feasable(customer):
                    vehicle.add_customer(customer)
                    assigned = True
                    break

            if not assigned:
                not_assigned.append(customer)

        print(f"Not Assigned: {len(not_assigned)}")

    def __call__(self, customers: List[Customer]) -> Tuple[List[Customer], float]:
        full_route = []
        fitness = 0
        for particle in self.particles:
            particle.load = 0
            particle.route = []
            particle.current_time = 0

        customer_priority = self.get_customer_priority_list(customers) # n-dimention
        priority_matrix = self.get_vehicle_priority_matrix(customer_priority) # 2m-dimention.
        self.construct_routes(priority_matrix)

        for idx, particle in enumerate(self.particles):
            route = particle.order_route(DISTANCE_MATRIX)
            for r in route:
                full_route.append(r)
            fitness += particle.evaluate(cost_function(route))

        return full_route, fitness

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

def get_local_best(customers: List[Customer], particles: List[Vehicle], index, k=5) -> Vehicle:
    neighbors = sorted(particles, key=lambda p: np.linalg.norm(particles[index].pos - p.pos))[:k]
    return min(neighbors, key=lambda p: p.evaluate(cost_function(customers)))

def get_neighbor_best(customers: List[Customer], particles: List[Vehicle], index) -> Vehicle:
    particle = particles[index]
    fdr_scores = [
        (np.abs(p.evaluate_pbest(cost_function(customers)) - particle.evaluate(cost_function(customers)))
         / np.linalg.norm(p.pos - particle.pos), p)
        for p in particles if p != particle
    ]
    best_neighbor = min(fdr_scores, key=lambda x: x[0])[1]
    return best_neighbor

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'c101.csv')
    customers = pd.read_csv(DATA_PATH)

    N_ITER = 150
    N_VEHICLES = 15
    N_PARTICLES = 50
    VEHICLE_CAPACITY = 200
    WIDTH, HEIGHT = get_search_space(customers)

    customers = Customer.create_from(customers)
    depot = customers[0]

    DISTANCE_MATRIX = get_distance_map(customers)
    customers = customers[1:]

    metrics = {
            'id': [],
            'max_fitness': [],
            'min_fitness': [],
            'avg_fitness': [],
            }

    gbest: Particle
    gbest_fitness = float('inf')
    gbest_route: List[Customer] = []
    for i in range(N_ITER):
        particles = [Particle(N_VEHICLES, (WIDTH, HEIGHT), depot) for _ in range(N_PARTICLES)]
        for pso in particles:
            route, fitness = pso(customers)
            if fitness < gbest_fitness:
                gbest = pso
                gbest_fitness = fitness
                gbest_route = route

            for idx, particle in enumerate(pso.particles):
                lbest = get_local_best(particle.route, pso.particles, idx)
                nbest = get_neighbor_best(particle.route, pso.particles, idx)
                particle.update_velocity(gbest.particles[idx], lbest, nbest)
                particle.update_position()
                particle.INERTIA = particle.INERTIA - ((particle.INERTIA - 0.4) * (i / N_ITER))

        print(f"Iteration {i+1}/{N_ITER}, Best Fitness: {gbest_fitness}")

    lngs, lats = zip(*[(c.lng, c.lat) for c in gbest_route])
    v_lngs, v_lats = zip(*[(p.pos[0], p.pos[1]) for p in gbest.particles])
    plt.plot(lngs, lats, '-o')
    plt.plot(v_lngs, v_lats, 'o', color='blue')
    plt.show()

