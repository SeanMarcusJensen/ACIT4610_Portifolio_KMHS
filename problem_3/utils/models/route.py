from .customer import Customer
import numpy as np

from dataclasses import dataclass


@dataclass
class Route:
    from_customer: Customer
    to_customer: Customer

    def distance(self) -> float:
        return np.sqrt((self.from_customer.lat - self.to_customer.lat) ** 2 +
                       (self.from_customer.lng - self.to_customer.lng) ** 2)

    def __str__(self) -> str:
        return f"Route({self.from_customer.customer_no} -> {self.to_customer.customer_no}[km: {self.distance}])"
