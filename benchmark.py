import time
import random

import settings
from settings import HERO, PLAYER_2, PLAYER_3, PLAYER_4, HeroPlayerSettings, StandardPlayerSettings
from monopoly.core.game import monopoly_game
from monopoly.log_settings import LogSettings

def benchmark_single_game():
    LogSettings.init_logs()  # Initialize logs so no file spam

    # Use a fixed seed for reproducibility
    seed = random.getrandbits(32)
    game_number = 1

    # Use default HeroPlayerSettings for simplicity
    settings.GameSettings.players_list = [
        (HERO, HeroPlayerSettings),
        (PLAYER_2, StandardPlayerSettings),
        (PLAYER_3, StandardPlayerSettings),
        (PLAYER_4, StandardPlayerSettings),
    ]

    start = time.time()
    summary = monopoly_game((game_number, seed))
    end = time.time()

    elapsed = end - start
    print(f"Single game run time: {elapsed:.4f} seconds")

if __name__ == "__main__":
    benchmark_single_game()
