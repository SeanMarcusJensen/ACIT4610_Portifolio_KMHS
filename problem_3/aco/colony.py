from utils.models import Customer, Route, RunningSheet
from typing import List
from utils.models import RunningSheet, Customer, Route
from utils.log import LoggerFactory

import numpy as np
from numpy.typing import NDArray
from typing import List, Callable


class Vehicle:
    def __init__(self, capacity: float, sheet: RunningSheet) -> None:
        self.__capacity = capacity
        self.__current_load: float = 0.0
        self.__current_time: float = 0.0
        self.__current_customer: Customer = sheet.depot
        self.__route: List[Route] = []
        self.__sheet = sheet

        self.__time_violations = 0

    def get_route(self) -> List[Route]:
        return self.__route

    def is_full(self) -> bool:
        return self.__current_load >= self.__capacity

    def is_empty(self) -> bool:
        return self.__current_load == 0.0

    def can_load(self, customer: Customer, shift_end: float) -> bool:
        can_do_load = self.__current_load + customer.demand <= self.__capacity
        can_do_in_time = self.__current_time + customer.service_time + \
            self.__sheet.get_distance_table()[self.__current_customer.get_id(), customer.get_id()] + \
            customer.service_time <= shift_end
        return can_do_load and can_do_in_time

    def get_current_customer_id(self) -> int:
        return self.__current_customer.get_id()

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

        table = self.__sheet.get_distance_table()
        distance = table[self.__current_customer.get_id(), customer.get_id()]

        route = Route(self.__current_customer, customer, distance)

        if (self.__current_time + distance) < customer.ready_time:
            self.__current_time = customer.ready_time
        else:
            self.__current_time += (distance + customer.service_time)

        if self.__current_time > customer.due_time:
            self.__time_violations += 1

        self.__route.append(route)
        customer.complete()
        self.__current_customer = customer

    def drive_route(self, unload_at_customer: Callable[[Route], None]) -> None:
        for route in self.__route:
            unload_at_customer(route)

    def total_time_violations(self) -> int:
        return self.__time_violations

    def total_distance(self) -> float:
        return sum(route.distance for route in self.__route)


class Colony:
    def __init__(self, sheet: RunningSheet, lf: LoggerFactory) -> None:
        self.__running_sheet = sheet
        self.__logger = lf.get_logger("Colony")
        self.__logger.info("Creating colony object.")
        self.__distance_table = self.__running_sheet.get_distance_table()

    def optimize(self, N_ITER: int, N_VEHICLES: int, CAPACITY: float, TAU: float, ALPHA: float, BETA: float, Q: float, P: float) -> List[Vehicle]:
        assert N_ITER > 0, "Number of iterations must be greater than 0."
        assert N_VEHICLES > 0, "Number of vehicles must be greater than 0."
        assert CAPACITY > 0.0, "Vehicle capacity must be greater than 0.0."
        assert TAU > 0.0, "Initial pheromone value must be greater than 0.0."
        assert 0.0 <= ALPHA <= 1.0, "Alpha must be between 0.0 and 1.0."
        assert 0.0 <= BETA <= 1.0, "Beta must be between 0.0 and 1.0."
        assert 0.0 <= Q <= 1.0, "Q must be between 0.0 and 1.0."
        assert 0.0 <= P <= 1.0, "P must be between 0.0 and 1.0."

        self.__logger.info("Optimizing the running sheet.")

        pheromones = self.__initialize_pheromones(N_ITER, TAU)

        vehicles: List[Vehicle] = []

        for t in range(N_ITER):
            self.__logger.debug(f"Starting Iteration({t + 1}/{N_ITER}).")

            # Reset the customers for the next iteration.
            self.__running_sheet.reset()

            # TODO: Update the pheromones.
            # [Ref: Lecture 6: Page 31] tau_ij (t + 1) ← (1 ⎯ P).tau_ij (t); 1 ≤ P ≤ 0
            # Doing it before because of convenience.
            pheromones[:, :, t + 1] = (1 - P) * pheromones[:, :, t]

            vehicles = [Vehicle(CAPACITY, self.__running_sheet)
                        for _ in range(N_VEHICLES)]

            while not self.__running_sheet.all_completed():
                for vehicle in vehicles:
                    if self.__running_sheet.all_completed():
                        break

                    # Start with random initial customer.
                    # TODO: Might want to change this to pick the customer with the earliest ready time.
                    available_customers = self.__running_sheet.get_remaining_customers()
                    next_customer = available_customers[np.random.choice(
                        len(available_customers))]

                    vehicle.load(next_customer)

                    while not vehicle.is_full():
                        # The vechicle will load up all the customers until it is full.
                        next_customer = self.__get_next_customer(
                            vehicle, pheromones, ALPHA, BETA, t)

                        if next_customer is None:
                            # Then there are no more customers to load.
                            break

                        vehicle.load(next_customer)

                    vehicle.load(self.__running_sheet.depot)

                    # Update phromones on the vehicle route.
                    # [Ref: Lecture 6: Page 34] tau_ij (t + 1) ← tau_ij (t + 1) + Q / L_k(t); 1 ≤ Q ≤ 0
                    # Only need to add the last part of the equation because the first part is already done.
                    def update_pheromones(route: Route) -> None:
                        pheromones[int(route.from_customer.get_id()), int(route.to_customer.get_id(
                        )), int(t + 1)] += (Q / route.distance)

                    vehicle.drive_route(update_pheromones)

        # Get the vehicle with the minimum distance.

        return vehicles

    def __get_next_customer(self, vehicle: Vehicle, pheromones: np.ndarray, ALPHA: float, BETA: float, t: int) -> Customer | None:
        available_customers = self.__running_sheet.get_remaining_customers()

        self.__logger.debug(
            f"Available customers: {len(available_customers)}.")

        allowed_customers = [
            customer for customer in available_customers if vehicle.can_load(customer, self.__running_sheet.depot.due_time)]

        if len(allowed_customers) <= 0:
            self.__logger.info(
                f"No customers available for vehicle.")
            return None

        if len(allowed_customers) == 1:
            self.__logger.info(
                f"Only one customer available for vehicle.")
            return allowed_customers[0]

        def calculate_prob(distance_table: np.ndarray, i: int, j: int, violations: int) -> float:
            self.__logger.debug(
                f"Calculating probability for {i} -> {j}. shape({pheromones.shape}), dtype: {pheromones.dtype}.")
            self.__logger.debug(
                f"Distance table shape: {distance_table.shape}, d={distance_table.dtype}.")
            objective = distance_table[int(i), int(j)] ** BETA
            nominator = (pheromones[int(i), int(
                j), int(t)] ** ALPHA) * objective
            denominator = sum((pheromones[int(i), int(k.get_id(
            )), int(t)] ** ALPHA) * self.__distance_table[int(i), int(k.get_id())] ** BETA for k in allowed_customers)
            return nominator / denominator

        probabilities = np.array([calculate_prob(self.__distance_table,
                                                 vehicle.get_current_customer_id(), j.get_id(), vehicle.total_time_violations()) for j in allowed_customers])

        self.__logger.debug(f"Probabilities: {probabilities}")

        cumsum = np.cumsum(probabilities)
        random_number = np.random.rand()

        node = - 1
        for index, cumulative_prob in enumerate(cumsum):
            if random_number <= cumulative_prob:
                node = index

        assert node != -1, "No node was selected."

        return allowed_customers[node]

    def __initialize_pheromones(self, t: int, TAU: float) -> NDArray[np.float64]:
        """ Creates a matrix of pheromones for the colony.

        Args:
            t (int): The number of iterations to run the opimization.

        Returns:
            NDArray[np.float64]: A numpy array of pheromones in shape (num_customers, num_customers, num_iterations) with initial value of TAU.
        """
        customers = [self.__running_sheet.depot] + \
            self.__running_sheet.customers

        shape = (len(customers), len(customers), t + 1)

        self.__logger.debug(f"Initializing pheromones with shape {shape}.")

        return np.full(shape, TAU)
