from .customer import Customer
import numpy as np

from dataclasses import dataclass


@dataclass
class Route:
    from_customer: Customer
    to_customer: Customer
    distance: float

    @staticmethod
    def calculate_distance(from_c: Customer, to_c: Customer) -> float:
        """ Calculates the distance between two customers.
        Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

        Returns:
            float: The distance between the two customers.
        """
        return np.sqrt((from_c.lat - to_c.lat) ** 2 +
                       (from_c.lng - to_c.lng) ** 2)

    def get_time_of_arrival(self, current_time: float) -> float:
        """ Calculates the time of arrival to the next customer from the current customer.

        Args:
            current_time (float): The time at which the current customer is being serviced.

        Returns:
            float: The time at which the vehicle will arrive at the next customer.
        """
        return current_time + self.distance

    def get_time_of_completion(self, current_time: float) -> float:
        """ Calculates the time of completion of the next customer from the current customer.

        Args:
            current_time (float): The time at which the current customer is being serviced.

        Returns:
            float: The time at which the vehicle will complete the service of the next customer.
        """
        return self.get_time_of_arrival(current_time) + self.to_customer.service_time

    def __str__(self) -> str:
        return f"Route({self.from_customer.customer_no} -> {self.to_customer.customer_no}[km: {self.distance}])"
