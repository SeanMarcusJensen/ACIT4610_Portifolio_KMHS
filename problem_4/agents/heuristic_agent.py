from .abstraction import TaxiAgent
from utils import EpsilonGreedy

class HeuristicPolicyAgent(TaxiAgent):
    def __init__(self) -> None:
        super().__init__(EpsilonGreedy(0.0, 0.0, 0.0))
            
        """ Walls are represented as a dictionary with key as a tuple of two points and value as a boolean. """
        self.__walls = {
            ((0, 0), (0, 1)): True,
            ((0, 1), (0, 0)): True,
            ((0, 3), (0, 4)): True,
            ((0, 4), (0, 3)): True,
            ((1, 0), (2, 0)): True,
            ((2, 0), (1, 0)): True,
            ((3, 0), (3, 1)): True,
            ((3, 1), (3, 0)): True,
            ((4, 0), (4, 1)): True,
            ((4, 1), (4, 0)): True,
        }

        self.__stuck_counter = 0

    # Function to check if a move is blocked
    def is_move_blocked(self, current_position: tuple, next_position: tuple) -> bool:
        return self.__walls.get((current_position, next_position), False)

    def _load(self) -> None:
        return
    
    def _save(self) -> None:
        return
    
    def _get_action(self, state: int) -> int:
        taxi_row, taxi_col, passenger_loc, dest_loc = self.__decode_state(state)
        """ passenger_loc & dest_loc is a number between 0-3; needs to be decoded to x,y coords """

        if passenger_loc < 4:
            return self.__handle_pickup(taxi_row, taxi_col, self._locations[passenger_loc])

        return self.__handle_dropoff(taxi_row, taxi_col, self._locations[dest_loc])
    
    def _update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        """ No training is needed, because the agent is deterministic.
        Might need to use prev state to make sure the agent is not stuck.
        """
        return

    def __decode_state(self, state: int) -> tuple:
        taxi_row, taxi_col, passenger_loc, dest_loc = self._env.unwrapped.decode(state) # type: ignore
        return taxi_row, taxi_col, passenger_loc, dest_loc

    def __handle_pickup(self, taxi_row: int, taxi_col: int, passenger_loc: tuple) -> int:
        """
        Handles moving to the passenger location.
        """
        if (taxi_row, taxi_col) == passenger_loc:
            return self._actions["pickup"]
        return self.__move_towards(taxi_row, taxi_col, *passenger_loc)

    def __handle_dropoff(self, taxi_row: int, taxi_col: int, dest_loc: tuple) -> int:
        if (taxi_row, taxi_col) == dest_loc:
            return self._actions["dropoff"]
        return self.__move_towards(taxi_row, taxi_col, *dest_loc)

    def __move_towards(self, taxi_row: int, taxi_col: int, target_row: int, target_col: int) -> int:
        if taxi_row < target_row and not self.is_move_blocked((taxi_row, taxi_col), (taxi_row + 1, taxi_col)):
            return self._actions["down"]
        elif taxi_row > target_row and not self.is_move_blocked((taxi_row, taxi_col), (taxi_row - 1, taxi_col)):
            return self._actions["up"]
        elif taxi_col < target_col and not self.is_move_blocked((taxi_row, taxi_col), (taxi_row, taxi_col + 1)):
            return self._actions["right"]
        elif taxi_col > target_col and not self.is_move_blocked((taxi_row, taxi_col), (taxi_row, taxi_col - 1)):
            return self._actions["left"]
        else:
            self.__stuck_counter += 1
            action = self.__stuck_counter % 4 # Always move to the right 
            return action


if __name__ == "__main__":
    heuristic_policy = HeuristicPolicyAgent()
    metrics = heuristic_policy.train(10000, 1000)
    metrics.plot(None)
    heuristic_policy.watch(2, 30)
