from .customer import Customer
import numpy as np

from dataclasses import dataclass


@dataclass
class Route:
    from_customer: Customer
    to_customer: Customer

    def distance(self) -> float:
        """ Calculates the distance between two customers.
        Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)

        Returns:
            float: The distance between the two customers.
        """
        return np.sqrt((self.from_customer.lat - self.to_customer.lat) ** 2 +
                       (self.from_customer.lng - self.to_customer.lng) ** 2)

    def get_time_of_arrival(self, current_time: float) -> float:
        """ Calculates the time of arrival to the next customer from the current customer.

        Args:
            current_time (float): The time at which the current customer is being serviced.

        Returns:
            float: The time at which the vehicle will arrive at the next customer.
        """
        return current_time + self.distance()

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
