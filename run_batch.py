# run_batch.py
import csv
import random
from typing import Dict, List, Tuple

import settings
from settings import HERO, PLAYER_2, PLAYER_3, PLAYER_4
from settings import HeroPlayerSettings, StandardPlayerSettings, SimulationSettings
from monopoly.core.game import monopoly_game
from monopoly.log_settings import LogSettings
from monopoly.core.constants import INDIGO, GREEN, YELLOW, RED, ORANGE, PINK, LIGHTBLUE, BROWN, RAILROADS, UTILITIES

# ---------- Experiment size (tweak to taste) ----------
N_STRATEGIES = 5          # number of distinct strategies to sample
N_GAMES_PER_STRATEGY = 1    # number of games simulated per strategy

# ---------- Parameter choices to sample from ----------
BOOL = [True, False]
MAX_DEV_OPTIONS = [3, 5]                 # 3 -> stop at 3 houses, 5 -> allow hotel
UNSPENDABLE_OPTIONS = [0, 50, 100, 200]
TRADE_MAX_DIFF_ABS = [0, 50, 100, 200]
SET_COMPLETION_BONUS = [0, 100, 200]

# Buy flags we will expose for the hero
BUY_FLAGS = [
    "buy_brown",
    "buy_light_blue",
    "buy_pink",
    "buy_orange",
    "buy_red",
    "buy_yellow",
    "buy_green",
    "buy_indigo",
    "buy_stations",
    "buy_utilities",
]

# ---------- Strategy sampler ----------
def sample_strategy(rng: random.Random) -> Dict:
    s = {}
    for flag in BUY_FLAGS:
        s[flag] = rng.choice(BOOL)
    s["max_development_level"] = rng.choice(MAX_DEV_OPTIONS)
    s["unspendable_cash"] = rng.choice(UNSPENDABLE_OPTIONS)
    s["trade_max_diff_absolute"] = rng.choice(TRADE_MAX_DIFF_ABS)
    s["set_completion_trade_bonus"] = rng.choice(SET_COMPLETION_BONUS)
    return s


def generate_distinct_strategies(n: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    seen = set()
    strategies = []
    while len(strategies) < n:
        s = sample_strategy(rng)
        # Make dedup key deterministic
        key = tuple((k, s[k]) for k in sorted(s.keys()))
        if key in seen:
            continue
        seen.add(key)
        strategies.append(s)
    return strategies


# ---------- Helpers to map boolean buy flags into ignored group names ----------
def build_ignored_groups(strategy: Dict) -> frozenset:
    """
    Convert the buy_* booleans into the board group constants that should be ignored.
    Uses constants from monopoly.core.constants to avoid typos and keep one source of truth.
    """
    ignored = set()
    if not strategy.get("buy_brown", True):
        ignored.add(BROWN)
    if not strategy.get("buy_light_blue", True):
        ignored.add(LIGHTBLUE)
    if not strategy.get("buy_pink", True):
        ignored.add(PINK)
    if not strategy.get("buy_orange", True):
        ignored.add(ORANGE)
    if not strategy.get("buy_red", True):
        ignored.add(RED)
    if not strategy.get("buy_yellow", True):
        ignored.add(YELLOW)
    if not strategy.get("buy_green", True):
        ignored.add(GREEN)
    if not strategy.get("buy_indigo", True):
        ignored.add(INDIGO)

    # stations / utilities
    if not strategy.get("buy_stations", True):
        ignored.add(RAILROADS)
    if not strategy.get("buy_utilities", True):
        ignored.add(UTILITIES)

    return frozenset(ignored)


# ---------- Main runner ----------
def run_single_game_with_strategy(strategy: Dict, game_number: int, seed: int) -> Dict:
    """
    Configure GameSettings so Hero uses the given strategy, run a single game, and return a result dict.
    The simulator expects GameSettings.players_list entries to be classes with attributes accessible at class level,
    so we create a subclass of HeroPlayerSettings with class attributes = our params.
    """
    # Build ignore set and params
    ignored = build_ignored_groups(strategy)
    params = {
        # pass fields HeroPlayerSettings expects at class-level
        "ignore_property_groups": ignored,
        "max_development_level": strategy["max_development_level"],
        "unspendable_cash": strategy["unspendable_cash"],
        "trade_max_diff_absolute": strategy["trade_max_diff_absolute"],
        "set_completion_trade_bonus": strategy["set_completion_trade_bonus"],
    }
    # also set the boolean buy flags as class attributes
    for flag in BUY_FLAGS:
        params[flag] = strategy[flag]

    # Dynamically create a subclass with these attributes as class attributes.
    # This way setup_players (which passes the class object) can find attributes as class-level defaults.
    CustomHeroSettings = type("CustomHeroSettings", (HeroPlayerSettings,), params)

    # Patch the simulator players list to use our custom class for HERO
    settings.GameSettings.players_list = [
        (HERO, CustomHeroSettings),
        (PLAYER_2, StandardPlayerSettings),
        (PLAYER_3, StandardPlayerSettings),
        (PLAYER_4, StandardPlayerSettings),
    ]

    # Run the game
    summary = monopoly_game((game_number, seed))

    # Defensive handling if summary is None / missing
    if summary is None:
        return {
            **strategy,
            "roi": 0.0,
            "win": 0,
            "props": 0,
            "houses": 0,
            "hotels": 0,
            "turns": SimulationSettings.n_moves,
        }

    hero_stats = summary.get("players", {}).get(HERO, {})
    return {
        **strategy,
        "roi": hero_stats.get("roi", 0.0),
        "win": 1 if summary.get("winner") == HERO else 0,
        "props": hero_stats.get("props", 0),
        "houses": hero_stats.get("houses", 0),
        "hotels": hero_stats.get("hotels", 0),
        "turns": hero_stats.get("turns", SimulationSettings.n_moves),
    }


def main():
    # Initialize logs (overwrite from previous runs)
    LogSettings.init_logs()

    rng = random.Random(SimulationSettings.seed)

    # Build strategies to test
    strategies = generate_distinct_strategies(N_STRATEGIES, seed=SimulationSettings.seed)
    print(f"Generated {len(strategies)} strategies. Running {N_GAMES_PER_STRATEGY} games per strategy -> total {len(strategies)*N_GAMES_PER_STRATEGY} games")

    # CSV header: buys + other hero params + outputs
    fieldnames = (
        BUY_FLAGS +
        ["max_development_level", "unspendable_cash", "trade_max_diff_absolute", "set_completion_trade_bonus"] +
        ["roi", "win", "props", "houses", "hotels", "turns"]
    )

    with open("strategy_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        game_counter = 0
        for strategy in strategies:
            print(f"Testing strategy: {strategy}")
            for _ in range(N_GAMES_PER_STRATEGY):
                game_counter += 1
                seed = rng.getrandbits(32)
                result = run_single_game_with_strategy(strategy, game_counter, seed)
                # ensure CSV only contains the expected columns
                out = {k: result.get(k, "") for k in fieldnames}
                writer.writerow(out)

    print("Done! Results saved to strategy_results.csv")


if __name__ == "__main__":
    main()
