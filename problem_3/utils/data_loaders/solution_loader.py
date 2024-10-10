import numpy as np
from typing import Dict

def read_solution(path: str) -> Dict[str, np.ndarray]:
    # Read the file
    with open(path, 'r') as f:
        lines = f.readlines()

    # Remove all boilerplate text
    routes = list(filter(lambda x: x.startswith("Route"), lines))

    # Remove '\n' from the end of each line
    routes = list(map(lambda x: x.strip(), routes))
    routes = {route.split(":")[0].strip("Route "): np.array(route.split(":")[1].split(), dtype=int) for route in routes}
    return routes