import pandas as pd
import numpy as np


class Customer:
    def __init__(self,
                 customer_no: int, demand: float,
                 lat: float, lng: float,
                 ready_time: float, due_time: float,
                 service_time: float):
        self.customer_no: int = int(customer_no - 0)
        self.demand = demand
        self.lat = lat
        self.lng = lng
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time
        self.completed = False
        print(f"Customer {self.customer_no} created.")
        print(f"customer_no: {self.customer_no}, demand: {demand}, lat: {lat}, lng: {lng}, ready_time: {ready_time}, due_time: {due_time}, service_time: {service_time}")

    @staticmethod
    def from_series(series: pd.Series) -> 'Customer':
        return Customer(
            customer_no=series['CustomerNO'],
            demand=series['Demand'],
            lat=series['Lat'],
            lng=series['Lng'],
            ready_time=series['ReadyTime'],
            due_time=series['Due'],
            service_time=series['ServiceTime']
        )

    def complete(self) -> None:
        self.completed = True

    def is_completed(self) -> bool:
        return self.completed

    def __eq__(self, other: 'Customer') -> bool:
        return self.customer_no == other.customer_no

    def __repr__(self) -> str:
        return f"customer_no: {self.customer_no}, demand: {self.demand}, lat: {self.lat}, lng: {self.lng}, ready_time: {self.ready_time}, due_time: {self.due_time}, service_time: {self.service_time}"
