import csv
from itertools import product
import random

import settings
from settings import HERO, PLAYER_2, PLAYER_3, PLAYER_4
from settings import HeroPlayerSettings, StandardPlayerSettings, SimulationSettings
from monopoly.core.game import monopoly_game
from monopoly.log_settings import LogSettings

# Number of simulations per strategy combo
N_GAMES_PER_STRATEGY = 5

# Strategy parameter grid
STRATEGIES = list(product(
    [True, False],  # buy_orange
    [True, False],  # buy_light_blue
    [True, False],  # buy_stations
    [True, False],  # buy_utilities
    [3, 5],          # max_development_level (3 houses only vs. up to hotel)
    [0, 100, 200],            # unspendable_cash
    [0, 100, 200]            # trade_max_diff_absolute
))


def run_single_game(strategy: dict, game_number: int, game_seed: int) -> dict:
    """
    Run one game with a given Hero strategy, return a dict with strategy + roi + win.
    """
    # Build ignored groups and params for HeroPlayerSettings
    ignored = set()
    if not strategy["buy_orange"]:
        ignored.add("ORANGE")
    if not strategy["buy_light_blue"]:
        ignored.add("LIGHT_BLUE")
    if not strategy["buy_stations"]:
        ignored.add("STATION")
    if not strategy["buy_utilities"]:
        ignored.add("UTILITY")

    params = {
        "buy_orange": strategy["buy_orange"],
        "buy_light_blue": strategy["buy_light_blue"],
        "buy_stations": strategy["buy_stations"],
        "buy_utilities": strategy["buy_utilities"],
        "max_development_level": strategy["max_development_level"],
        "ignore_property_groups": frozenset(ignored),
        "unspendable_cash": strategy["unspendable_cash"],
        "trade_max_diff_absolute": strategy["trade_max_diff_absolute"],
    }

    # Dynamically create a Settings class that hardcodes these params
    CustomHeroSettings = type(
        "CustomHeroSettings",
        (HeroPlayerSettings,),
        {
            "__init__": lambda self: super(CustomHeroSettings, self).__init__(**params)
        }
    )

    # Patch the simulator’s players_list to use our custom HeroSettings
    settings.GameSettings.players_list = [
        (HERO, CustomHeroSettings),
        (PLAYER_2, StandardPlayerSettings),
        (PLAYER_3, StandardPlayerSettings),
        (PLAYER_4, StandardPlayerSettings),
    ]

    # Run the game
    summary = monopoly_game((game_number, game_seed))

    # Extract Hero stats
    hero_stats = summary["players"][HERO]
    return {
        **strategy,
        "roi": hero_stats["roi"],
        "win": 1 if summary["winner"] == HERO else 0,
        "props":  hero_stats["props"],
        "houses": hero_stats["houses"],
        "hotels": hero_stats["hotels"],
        "turns":  hero_stats["turns"]
    }


if __name__ == "__main__":
    # Initialize logs (so simulator doesn’t spam files)
    LogSettings.init_logs()

    # Prepare CSV
    with open("strategy_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "buy_orange", "buy_light_blue", "buy_stations", "buy_utilities",
            "max_development_level", "unspendable_cash", "trade_max_diff_absolute",
            "roi", "win",
            "props", "houses", "hotels", "turns"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Base RNG for consistent seeds
        rng = random.Random(SimulationSettings.seed)
        game_counter = 0

        # Loop over each strategy combo
        for combo in STRATEGIES:
            strategy = {
                "buy_orange": combo[0],
                "buy_light_blue": combo[1],
                "buy_stations": combo[2],
                "buy_utilities": combo[3],
                "max_development_level": combo[4],
                "unspendable_cash": combo[5],
                "trade_max_diff_absolute": combo[6],
            }
            print(f"Testing strategy: {strategy}")

            # Run multiple games per strategy
            for _ in range(N_GAMES_PER_STRATEGY):
                game_counter += 1
                seed = random.getrandbits(32)
                result = run_single_game(strategy, game_counter, seed)
                writer.writerow(result)

    print("Done! Results saved to strategy_results.csv")
