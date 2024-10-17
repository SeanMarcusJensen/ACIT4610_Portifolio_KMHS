
import pandas as pd
import numpy as np
from typing import List, Callable
from dataclasses import dataclass


class Customer:
    def __init__(self, id: float, customer_no: float, x: float, y: float, demand: float, ready_time: float, due_time: float, service_time: float) -> None:
        self.id: int = int(id)
        self.customer_no = customer_no
        self.x: float = x
        self.y: float = y
        self.demand: float = demand
        self.ready_time: float = ready_time
        self.due_time: float = due_time
        self.service_time: float = service_time

        self.__is_serviced = False

    def set_serviced(self) -> None:
        self.__is_serviced = True

    def is_serviced(self) -> bool:
        return self.__is_serviced

    def reset(self) -> None:
        self.__is_serviced = False

    def __str__(self) -> str:
        return f"Customer(no: {self.customer_no}) demand: {self.demand} ready: {self.ready_time} due: {self.due_time} service: {self.service_time}"


class RunningSheet:
    def __init__(self, customers: List[Customer]) -> None:
        self.__customers: List[Customer] = customers[1:]
        self.__depot: Customer = customers[0]
        self.distance_table: np.ndarray = self.__create_distance_table()

    def create_copy(self) -> 'RunningSheet':
        customers = [self.__depot] + self.__customers
        return RunningSheet(customers.copy())

    def get_travel_time(self, from_: Customer, to_: Customer) -> float:
        return self.distance_table[from_.id, to_.id]

    def get_unserviced_customers(self) -> List[Customer]:
        return [c for c in self.__customers if not c.is_serviced()]

    def is_completed(self) -> bool:
        return all([c.is_serviced() for c in self.__customers])

    def get_customers(self) -> List[Customer]:
        return self.__customers

    def get_depot(self) -> Customer:
        return self.__depot

    def reset(self) -> None:
        for c in self.__customers:
            c.reset()

    def __create_distance_table(self) -> np.ndarray:
        """ Creates a distance table.

        Returns:
            np.ndarray: The distance table.
        """
        customers = [self.__depot] + self.__customers
        n = len(customers)
        distance_table = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x = customers[i]
                y = customers[j]
                distance_table[i, j] = np.sqrt((x.x - y.x)**2 + (x.y - y.y)**2)
        return distance_table

    @staticmethod
    def from_pd(locations: pd.DataFrame) -> 'RunningSheet':
        """ Creates a running sheet from a pandas DataFrame.

        Args:
            locations (pd.DataFrame): The locations DataFrame.

        Returns:
            RunningSheet: The running sheet.
        """
        customers = [Customer(*row) for _, row in locations.iterrows()]
        return RunningSheet(customers=customers)


@dataclass
class Route:
    from_: Customer
    to_: Customer
    distance: float

    def __repr__(self) -> str:
        return f"Route(from: {self.from_.customer_no}, to: {self.to_.customer_no})"


class Vehicle:
    def __init__(self, id: int, capacity: float, sheet: RunningSheet) -> None:
        self.__id: int = id
        self.__sheet = sheet
        self.__depot: Customer = sheet.get_depot()
        self.__routes: List[Route] = []
        self.__capacity: float = capacity
        self.__current_load: float = 0.0
        self.__current_time: float = 0.0
        self.__current_customer: Customer = self.__depot
        self.__current_customer.set_serviced()

        self.__time_violations = 0

    def reset(self) -> None:
        self.__current_load = 0.0
        self.__current_time = 0.0
        self.__current_customer = self.__depot
        self.__sheet.reset()
        self.__time_violations = 0
        self.__routes = []

    def get_route(self) -> List[Route]:
        return self.__routes

    def is_full(self) -> bool:
        return self.__current_load >= self.__capacity

    def get_current_customer(self) -> Customer:
        return self.__current_customer

    def can_load(self, customer: Customer) -> bool:
        can_do_load = self.__current_load + customer.demand <= self.__capacity
        can_do_in_time = self.__current_time + customer.service_time + \
            self.__sheet.get_travel_time(self.__current_customer, customer) + \
            customer.service_time <= self.return_time()
        return can_do_load and can_do_in_time

    def get_distance_table(self) -> np.ndarray:
        return self.__sheet.distance_table

    def return_time(self) -> float:
        return self.__depot.due_time

    def in_time_window(self, customer: Customer) -> bool:
        """ Checks if the vehicle can reach the customer within the time window.

        Args:
            customer (Customer): The customer.

        Returns:
            bool: True if the vehicle can reach the customer within the time window, False otherwise.
        """
        time_window = (self.__current_time + self.__sheet.get_travel_time(
            self.__current_customer, customer), self.return_time())

        return customer.ready_time <= time_window[0] and time_window[1] <= customer.due_time

    def total_travel_time(self) -> float:
        return sum([r.distance for r in self.__routes])

    def load(self, customer: Customer):
        """ Loads the customer to its route.
        If the customer is not ready, the vehicle will wait until the customer is ready.

        The customer is completed after loading to protect against double loading.

        Args:
            customer (Customer): The customer to be loaded.
        """
        if self.__current_customer is customer:
            print("Customer already loaded.")
            return

        self.__current_load += customer.demand

        route = Route(self.__current_customer, customer, self.__sheet.get_travel_time(
            self.__current_customer, customer))

        if (self.__current_time + route.distance) < customer.ready_time:
            self.__current_time = customer.ready_time
        else:
            self.__current_time += (route.distance + customer.service_time)

        if self.__current_time > customer.due_time:
            self.__time_violations += 1

        self.__routes.append(route)
        customer.set_serviced()
        self.__current_customer = customer

    def drive_route(self, unload_at_customer: Callable[[Route], None]):
        for route in self.__routes:
            unload_at_customer(route)

    def total_time_violations(self) -> int:
        return self.__time_violations

    def load_random_customer(self) -> None:
        available_customers = self.__sheet.get_unserviced_customers()

        random_starting_customer = available_customers[np.random.choice(
            len(available_customers))]

        self.load(random_starting_customer)

    def get_allowed_customers(self) -> List[Customer]:
        customers = self.__sheet.get_unserviced_customers()
        return [c for c in customers if self.can_load(c)]

    def has_shift_ended(self) -> bool:
        return self.__current_time > self.return_time() or self.__sheet.is_completed()

    def head_back_to_depot(self) -> None:
        self.load(self.__depot)
        self.__current_load = 0.0

    def __repr__(self) -> str:
        return f"Vehicle({self.__id}) handles {len(self.__routes)} customers and has traveled {self.total_travel_time()} units."


class Colony:
    def __init__(self, sheet: RunningSheet) -> None:
        self.__running_sheet: RunningSheet = sheet

    def optimize(self, N_ITER: int, N_VEHICLES: int, CAPACITY: float, TAU: float, ALPHA: float, BETA: float, Q: float, P: float) -> List[Vehicle]:
        assert N_ITER > 0, "Number of iterations must be greater than 0."
        assert N_VEHICLES > 0, "Number of vehicles must be greater than 0."
        assert CAPACITY > 0.0, "Vehicle capacity must be greater than 0.0."
        assert TAU > 0.0, "Initial pheromone value must be greater than 0.0."
        assert 0.0 <= ALPHA <= 1.0, "Alpha must be between 0.0 and 1.0."
        assert 0.0 <= BETA <= 1.0, "Beta must be between 0.0 and 1.0."
        assert 0.0 <= Q <= 1.0, "Q must be between 0.0 and 1.0."
        assert 0.0 <= P <= 1.0, "P must be between 0.0 and 1.0."

        pheromones = self.__initialize_pheromones(N_ITER, TAU)
        vehicles = [Vehicle(i, CAPACITY, self.__running_sheet.create_copy())
                    for i in range(N_VEHICLES)]

        # Training loop
        for t in range(N_ITER):

            # Evaporate the pheromones for next iteration.
            pheromones[:, :, t + 1] = (1 - P) * pheromones[:, :, t]

            for vehicle in vehicles:
                # Resets the vehicle, the sheet and the time violations.
                vehicle.reset()

                # initialize the vehicle with a random customer.
                vehicle.load_random_customer()

                # Shift ends when time is after depot time or all customers are serviced.
                # Might want to do all customers either way, but just give them violations.
                while not vehicle.has_shift_ended():
                    if vehicle.is_full():
                        vehicle.head_back_to_depot()
                        continue

                    next_customer = self.__get_next_customer(
                        vehicle, pheromones, ALPHA, BETA, t)

                    if next_customer is None:
                        vehicle.head_back_to_depot()
                        break

                    vehicle.load(next_customer)

                def update_pheromones(route: Route) -> None:
                    pheromones[route.from_.id, route.to_.id,
                               t + 1] += (Q / route.distance)

                # Update pheromones on the route for the next iteration.
                vehicle.drive_route(update_pheromones)
        return vehicles

    def __get_next_customer(self, vehicle: Vehicle, pheromones: np.ndarray, ALPHA: float, BETA: float, iter: int) -> Customer | None:
        allowed_customers = vehicle.get_allowed_customers()

        if len(allowed_customers) <= 0:
            return None

        if len(allowed_customers) == 1:
            return allowed_customers[0]

        def calculate_probability(distance_table: np.ndarray, i: int, j: int) -> float:
            objective = distance_table[i, j] ** BETA
            nominator = (pheromones[i, j, iter] ** ALPHA) * objective
            denominator = sum((pheromones[i, k.id, iter] ** ALPHA) * (
                distance_table[i, k.id] ** BETA) for k in allowed_customers)
            return nominator / denominator

        probabilities = [calculate_probability(
            vehicle.get_distance_table(), vehicle.get_current_customer().id, c.id) for c in allowed_customers]

        cumsum = np.cumsum(probabilities)
        random_number = np.random.rand()

        node = -1
        for index, cumulative_prob in enumerate(cumsum):
            if random_number <= cumulative_prob:
                node = index

        assert node != -1, "No node was selected."

        return allowed_customers[node]

    def __initialize_pheromones(self, N_ITER: int, TAU: float) -> np.ndarray:
        customers = [self.__running_sheet.get_depot()] + \
            self.__running_sheet.get_customers()
        return np.full((len(customers), len(customers), N_ITER + 1), TAU)


if __name__ == "__main__":
    import pandas as pd
    import os
    path = os.path.dirname(__file__) + '/data/processed/c101.csv'

    locations = pd.read_csv(path)

    colony = Colony(RunningSheet.from_pd(locations))
    vehicles = colony.optimize(
        N_ITER=10, N_VEHICLES=10, CAPACITY=200, TAU=0.1, ALPHA=0.5, BETA=0.5, Q=0.1, P=0.1)

    for vehicle in vehicles:
        print(vehicle)
        print(vehicle.total_time_violations())
        print(vehicle.get_route())
        print()
