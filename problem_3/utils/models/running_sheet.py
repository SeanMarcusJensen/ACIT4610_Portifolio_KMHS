from .customer import Customer
from .route import Route
from utils.log import ILogger, LoggerFactory

import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List

from dataclasses import dataclass


@dataclass
class RunningSheet:
    depot: Customer
    customers: List[Customer]
    logger: ILogger

    @staticmethod
    def from_csv(abs_path: str, lf: LoggerFactory) -> 'RunningSheet':
        data = pd.read_csv(abs_path)
        return RunningSheet.from_pd(data, lf)

    @staticmethod
    def from_pd(data: pd.DataFrame, lf: LoggerFactory) -> 'RunningSheet':
        customers = [Customer.from_series(row, lf)
                     for _, row in data.iterrows()]
        depot = customers.pop(0)
        logger = lf.get_logger("RunningSheet")
        logger.info(
            f"Creating RunningSheet object with [{len(customers)}] customers.")
        return RunningSheet(depot, customers, logger)

    def get_distance_table(self) -> npt.NDArray[np.float64]:
        customers = [self.depot] + self.customers
        return np.array([[Route(customer, other).distance() for other in customers] for customer in customers])

    def get_remaining_customers(self) -> List[Customer]:
        total_customers = len(self.customers)
        uncompleted_customers = [
            customer for customer in self.customers if not customer.is_completed()]
        self.logger.debug(
            f"Completed customers: {total_customers - len(uncompleted_customers)}/{total_customers}.")
        return uncompleted_customers

    def reset(self) -> None:
        self.logger.debug(
            "Resetting all customers, depot is staying as completed.")
        for customer in self.customers:
            customer.completed = False

    def all_completed(self) -> bool:
        self.logger.debug(
            f"Checking if all customers are completed.")
        return all(customer.is_completed() for customer in self.customers)

    def __iter__(self):
        return iter(self.customers)

    def __getitem__(self, val: int) -> Customer:
        return self.customers[val]
