from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Dict

DISTANCE_MATRIX = None

class Customer:
    """Represents a customer in a vehicle routing problem with time windows (VRPTW).
    
    Attributes:
        customer_no (int): Unique identifier for the customer.
        lng (int): Longitude or x-coordinate of the customer's location.
        lat (int): Latitude or y-coordinate of the customer's location.
        demand (int): Demands for each customer.
        ready_time (int): The earliest time the customer is ready for service.
        due (int): The due time for start of service.
        service_time (int): Time it takes to serve each customer.
    """
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
    """Represents a vehicle in the Particle Swarm Optimization (PSO) solution for VRPTW, handling movement and
    service constraints.
    
    Attributes:
        CAPACITY (int): The maximum capacity each vehicle can carry.
        INERTIA, COGNITIVE, SOCIAL, LOCAL, NEIGHBOR (float): Constants for velocity updates in PSO.
        id (int): Unique identifier for the vehicle.
        bounds (tuple): Spatial boundaries for vehicle position.
        pos (np.array): Current position of the vehicle in the search space.
        velocity (np.array): Current velocity of the vehicle in the search space.
        velocity_cap (tuple): Bounds on the maximum velocity.
        route (List[Customer]): Ordered list of customers assigned to the vehicle.
        pbest (np.array): Best-known position of the vehicle in the solution space.
        load (float): Current load of the vehicle.
        current_time (float): Current time in the vehicle's route schedule.
        depot (Customer): The depot, serving as the start and end of the route.
    """
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

    def copy(self) -> 'Vehicle':
        particle = Vehicle(self.id, self.depot, self.bounds[0], self.bounds[1])
        particle.pos = self.pos.copy()
        particle.velocity = self.velocity.copy()
        particle.pbest = self.pbest.copy()
        particle.route = self.route.copy()
        particle.load = self.load
        particle.current_time = self.current_time
        return particle

    def add_customer(self, customer: Customer) -> None:
        """ Update Load, Current Time etc."""
        self.load += customer.demand

        current_customer = self.route[-1] if len(self.route) > 0 else self.depot

        travel_time = calculate_distance(current_customer.lng, customer.lng, current_customer.lat, customer.lat)
        arrival_time = self.current_time + travel_time
        ready_time, due_time = customer.ready_time, customer.due

        if arrival_time < ready_time:
            self.current_time = ready_time + customer.service_time
        else:
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
        
        return [self.depot] + self.route + [self.depot]  # Return route with depot at start and end

    def distance_to_customer(self, customer: Customer) -> float:
        return np.sqrt((self.pos[0] - customer.lng) ** 2 + (self.pos[1] - customer.lat) ** 2)

    def is_customer_feasable(self, customer: Customer) -> bool:
        if self.load + customer.demand > self.CAPACITY:
            return False

        # If not in time for the due, return False
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
    """Represents a set of vehicles, or "particles", for PSO in VRPTW. Manages vehicle assignments to customers,
    route construction, and solution evaluation.
    
    Attributes:
        particles (np.array): Array of Vehicle objects representing the swarm's particles.
        pbest (np.array): Array storing each vehicle's best-known position.
    """
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
    neighbors = sorted(particles, key=lambda p: np.linalg.norm(particles[index].pos - p.pos))[:k] # type: ignore
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

def set_distance_matrix(customers):
    global DISTANCE_MATRIX
    
    distance_matrix = get_distance_map(customers)

    DISTANCE_MATRIX = distance_matrix

def optimize(n_iterations, n_vehicles, n_particles, width, height, depot, customers, plot=False):    
    # Track convergence
    total_fitness_per_iteration = []

    gbest: Particle
    gbest_fitness = float('inf')
    gbest_route: List[Customer] = []
    best_vehicle_solutions = []
    best_fitness_index = 0

    for i in range(n_iterations):
        particles = [Particle(n_vehicles, (width, height), depot) for _ in range(n_particles)]
        for pso in particles:
            route, fitness = pso(customers)
            if fitness < gbest_fitness:
                gbest = pso
                gbest_fitness = fitness
                gbest_route = route
                best_fitness_index = i
                best_vehicle_solutions = [vehicle.route for vehicle in pso.particles]  # Save route per vehicle

            for idx, particle in enumerate(pso.particles):
                lbest = get_local_best(particle.route, pso.particles, idx) # type: ignore
                nbest = get_neighbor_best(particle.route, pso.particles, idx) # type: ignore
                particle.update_velocity(gbest.particles[idx], lbest, nbest)
                particle.update_position()
                particle.INERTIA = particle.INERTIA - ((particle.INERTIA - 0.4) * (i / n_iterations))

        total_fitness_per_iteration.append(gbest_fitness)

        print(f"Iteration {i+1}/{n_iterations}, Best Fitness: {gbest_fitness}")

    if plot:
        plot_pso_vehicle_routes(customers, gbest, n_vehicles, depot, best_vehicle_solutions)

    max_customers_visited = len([customer for route in best_vehicle_solutions for customer in route if customer != depot])

    return best_vehicle_solutions, max_customers_visited, gbest_fitness, total_fitness_per_iteration, best_fitness_index

def plot_pso_vehicle_routes(customers, gbest, n_vehicles, depot, best_vehicle_solutions):
    # Assign distinct colors to each vehicle route
    colors = plt.cm.get_cmap('tab20', n_vehicles)

    plt.figure(figsize=(10, 8))

    # Plot the depot
    plt.scatter([depot.lng], [depot.lat], color='red', label='Depot', s=200, zorder=3)

    # Plot the customer locations
    customer_lngs, customer_lats = zip(*[(c.lng, c.lat) for c in customers])
    plt.scatter(customer_lngs, customer_lats, color='blue', label='Customers')

    for vehicle_idx, route in enumerate(best_vehicle_solutions):
        if route:
            vehicle_lngs, vehicle_lats = zip(*[(c.lng, c.lat) for c in route])

            # Connect the depot to the first customer
            plt.plot([depot.lng, vehicle_lngs[0]], [depot.lat, vehicle_lats[0]], color=colors(vehicle_idx))

            # Plot the rest of the route
            plt.plot(vehicle_lngs, vehicle_lats, '-o', color=colors(vehicle_idx), label=f'Vehicle {vehicle_idx + 1}')

            # Connect the last customer back to the depot
            plt.plot([depot.lng, vehicle_lngs[-1]], [depot.lat, vehicle_lats[-1]], color=colors(vehicle_idx))

    # Plot the particle positions
    v_lngs, v_lats = zip(*[(p.pos[0], p.pos[1]) for p in gbest.particles])
    plt.scatter(v_lngs, v_lats, color='green', marker='x', label='Particle Positions', zorder=2)

    plt.title("Vehicle Routes for VRPTW Solution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(True)

    plt.show()

def print_pso_solutions(vehicle_solutions: List[List[Customer]], best_fitness: float, max_customers_visited: int, best_solution_index: float):
    print(f"Best PSO solution (Index: {best_solution_index + 1})")

    for i, solution in enumerate(vehicle_solutions):
        # Exclude the depot
        route_str = ' -> '.join([str(c.customer_no) for c in solution[1:-1]])
        print(f"Route {i + 1}: {route_str}")

    print(f"\nTotal number of customers visited: {max_customers_visited}/100")
    print(f"Total distance: {best_fitness}")

def plot_pso_convergence(total_fitness_per_iteration: List[float], n_iterations: int):
    plt.figure(figsize=(10, 8))
    # Plot the best fitness value per iteration
    plt.plot(range(1, len(total_fitness_per_iteration) + 1), total_fitness_per_iteration, marker='o')
    plt.title("PSO Convergence Plot - Best Fitness per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness Score")

    plt.xticks(range(0, n_iterations, 5))

    plt.grid(True)
    plt.show()