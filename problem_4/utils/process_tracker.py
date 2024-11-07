import datetime
from typing import Callable
import os


def create_progress_tracker(title: str | None = None) -> Callable[[int, int], None]:
    LENGTH = 50
    start_time: datetime.datetime | None = None
    os.system('cls' if os.name == 'nt' else 'clear')

    print("Training Progress" if title is None else title)

    def progress_tracker(episode: int, n_episodes: int) -> None:
        nonlocal start_time
        start_time = start_time or datetime.datetime.now()  # only set start_time once

        if episode % 100 == 0:
            elapsed_time = datetime.datetime.now() - start_time
            elapsed_time = elapsed_time.total_seconds()
            completion = episode / n_episodes
            fill = "-" * int(LENGTH * (1 - completion))
            bar = "#" * (LENGTH - len(fill))
            print(
                f"Episode: {episode}/{n_episodes}|{bar}{fill}|{(completion*100):.1f}% Complete. Elapsed {elapsed_time:.1f}s.", end='\r')

    return progress_tracker
