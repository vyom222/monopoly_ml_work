import csv
from itertools import product
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import settings
from settings import HERO, PLAYER_2, PLAYER_3, PLAYER_4
from settings import HeroPlayerSettings, StandardPlayerSettings, SimulationSettings
from monopoly.core.game import monopoly_game
from monopoly.log_settings import LogSettings
from monopoly.core.constants import (
    BROWN, LIGHTBLUE, PINK, ORANGE, RED, YELLOW, GREEN, INDIGO, RAILROADS, UTILITIES
)

N_GAMES_PER_STRATEGY = 100

# Full boolean buy flags for all groups
BUY_FLAGS = ['buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
             'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
             'buy_stations', 'buy_utilities']

STRATEGY_GRID = list(product(
    *[[True, False] for _ in BUY_FLAGS],
    [3, 5],                 # max_development_level
    [0, 50, 100],     # unspendable_cash
))

def run_single_game(strategy: dict, game_number: int, game_seed: int) -> dict:
    # Determine which groups to ignore based on buy flags
    ignored = set()
    group_map = {
        "buy_brown": BROWN,
        "buy_light_blue": LIGHTBLUE,
        "buy_pink": PINK,
        "buy_orange": ORANGE,
        "buy_red": RED,
        "buy_yellow": YELLOW,
        "buy_green": GREEN,
        "buy_indigo": INDIGO,
        "buy_stations": RAILROADS,
        "buy_utilities": UTILITIES,
    }
    for flag, group in group_map.items():
        if not strategy[flag]:
            ignored.add(group)

    params = {
        **{flag: strategy[flag] for flag in BUY_FLAGS},
        "max_development_level": strategy["max_development_level"],
        "ignore_property_groups": frozenset(ignored),
        "unspendable_cash": strategy["unspendable_cash"],
        "trade_max_diff_absolute": 150,
        "set_completion_trade_bonus": 250,
    }

    CustomHeroSettings = type(
        "CustomHeroSettings",
        (HeroPlayerSettings,),
        {
            "__init__": lambda self: super(CustomHeroSettings, self).__init__(**params)
        }
    )

    # Patch the simulator’s players list and seed (these are globals, so careful with multiprocessing!)
    settings.GameSettings.players_list = [
        (HERO, CustomHeroSettings),
        (PLAYER_2, StandardPlayerSettings),
        (PLAYER_3, StandardPlayerSettings),
        (PLAYER_4, StandardPlayerSettings),
    ]
    settings.GameSettings.seed = game_seed

    summary = monopoly_game((game_number, game_seed))

    hero_stats = summary["players"][HERO]
    return {
        **strategy,
        "roi": hero_stats["roi"],
        "win": 1 if summary["winner"] == HERO else 0,
        "props": hero_stats["props"],
        "houses": hero_stats["houses"],
        "hotels": hero_stats["hotels"],
        "turns": hero_stats["turns"],
        "game_seed": game_seed,
    }


if __name__ == "__main__":
    LogSettings.init_logs()

    # Prepare CSV
    with open("strategy_results.csv", "w", newline="") as csvfile:
        fieldnames = BUY_FLAGS + [
            "max_development_level", "unspendable_cash",
            "roi", "win", "props", "houses", "hotels", "turns", "game_seed"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        rng = random.Random(SimulationSettings.seed)
        game_counter = 0

        # Prepare all jobs to submit to the executor
        jobs = []
        for combo in STRATEGY_GRID:
            strategy = {flag: combo[i] for i, flag in enumerate(BUY_FLAGS)}
            strategy.update({
                "max_development_level": combo[len(BUY_FLAGS)],
                "unspendable_cash": combo[len(BUY_FLAGS) + 1],
            })
            for _ in range(N_GAMES_PER_STRATEGY):
                game_counter += 1
                seed = rng.getrandbits(32)
                jobs.append((strategy, game_counter, seed))

        # Run in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_single_game, strat, num, seed) for strat, num, seed in jobs]
            for future in as_completed(futures):
                result = future.result()
                writer.writerow(result)
                print(f"Completed game with seed {result['game_seed']} - win: {result['win']}")

    print("Done! Results saved to strategy_results.csv")
