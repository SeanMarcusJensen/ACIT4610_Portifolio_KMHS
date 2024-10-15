import pandas as pd
import numpy as np
from utils.log import ILogger, LoggerFactory
from dataclasses import dataclass


class Customer:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, customer_no: int, demand: float, lat: float, lng: float, ready_time: float, due_time: float, service_time: float, lf: LoggerFactory):
        self.customer_no: int = customer_no  # Customer number
        # Demand of the customer - how much packages they have.
        self.demand: float = demand
        self.lat: float = lat  # Latitude
        self.lng: float = lng  # Longitude
        # Time at which the customer is ready to be serviced.
        self.ready_time: float = ready_time
        # Time by which the customer must be serviced.
        self.due_time: float = due_time
        # Time it takes to service the customer.
        self.service_time: float = service_time
        # Whether the customer has been serviced or not.
        self.completed = False
        self.__logger: ILogger = lf.get_logger(f"Customer({self.customer_no})")

    @staticmethod
    def from_series(series: pd.Series, lf: LoggerFactory) -> 'Customer':
        return Customer(
            customer_no=series['CustomerNO'],
            demand=series['Demand'],
            lat=series['Lat'],
            lng=series['Lng'],
            ready_time=series['ReadyTime'],
            due_time=series['Due'],
            service_time=series['ServiceTime'],
            lf=lf
        )

    def get_id(self) -> int:
        id = self.customer_no - 1
        assert id >= 0, "Customer ID must be greater than or equal to 0."
        return id

    def complete(self) -> None:
        self.completed = True

    def is_completed(self) -> bool:
        return self.completed

    def __eq__(self, other: 'Customer') -> bool:
        return self.customer_no == other.customer_no

    def __repr__(self) -> str:
        return f"Customer({self.customer_no})[x: {self.lat}, y: {self.lng}]"
