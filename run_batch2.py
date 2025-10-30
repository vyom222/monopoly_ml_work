import csv
from itertools import product
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

import settings
from settings import HERO, PLAYER_2, PLAYER_3, PLAYER_4
from settings import HeroPlayerSettings, StandardPlayerSettings, SimulationSettings
from monopoly.core.game import monopoly_game
from monopoly.log_settings import LogSettings
from monopoly.core.constants import (
    BROWN, LIGHTBLUE, PINK, ORANGE, RED, YELLOW, GREEN, INDIGO, RAILROADS, UTILITIES
)

# Experiment parameters
N_GAMES_PER_STRATEGY = 100  
N_WORKERS = None  

BUY_FLAGS = ['buy_brown', 'buy_light_blue', 'buy_pink', 'buy_orange',
             'buy_red', 'buy_yellow', 'buy_green', 'buy_indigo',
             'buy_stations', 'buy_utilities']

STRATEGY_GRID = list(product(
    *[[True, False] for _ in BUY_FLAGS],
    [3, 5],                 # max_development_level
    [0, 50, 100],           # unspendable_cash
))

GROUP_ORDER = [BROWN, LIGHTBLUE, PINK, ORANGE, RED, YELLOW, GREEN, INDIGO, RAILROADS, UTILITIES]


def normalize_group_name(g):
    return str(g).lower().replace(" ", "_").replace("'", "").replace("-", "_")

GROUP_NAMES = [normalize_group_name(g) for g in GROUP_ORDER]

def _make_custom_hero_class(params: dict):
    """
    Create a subclass of HeroPlayerSettings with class attributes set to params.
    This avoids depending on __init__ signature of GameSettings elsewhere.
    """
    class_attrs = dict(params) 
    return type("CustomHeroSettings", (HeroPlayerSettings,), class_attrs)


def run_single_game(job):
    # Create a CustomHeroSettings class whose attributes match the strategy,
    strategy, game_number, game_seed = job

    # Determine which groups to ignore based on buy flags
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
    ignored = set(group_map[flag] for flag in group_map if not strategy.get(flag, True))

    params = {
        **{flag: strategy[flag] for flag in BUY_FLAGS},
        "max_development_level": strategy["max_development_level"],
        "ignore_property_groups": frozenset(ignored),
        "unspendable_cash": strategy["unspendable_cash"],
        # fixed trading parameters
        "trade_max_diff_absolute": getattr(strategy, "trade_max_diff_absolute", 150) if hasattr(strategy, "trade_max_diff_absolute") else 150,
        "set_completion_trade_bonus": getattr(strategy, "set_completion_trade_bonus", 250) if hasattr(strategy, "set_completion_trade_bonus") else 250,
    }

    CustomHeroSettings = _make_custom_hero_class(params)

    # Patch GameSettings
    settings.GameSettings.players_list = [
        (HERO, CustomHeroSettings),
        (PLAYER_2, StandardPlayerSettings),
        (PLAYER_3, StandardPlayerSettings),
        (PLAYER_4, StandardPlayerSettings),
    ]
    settings.GameSettings.seed = game_seed

    try:
        summary = monopoly_game((game_number, game_seed))
    except Exception as e:
        return {"error": True, "error_message": str(e), "strategy": strategy, "game_seed": game_seed}

    # Extract Hero stats
    hero_stats = summary["players"].get(HERO, {})

    # Build base result 
    result = {k: strategy.get(k, "") for k in BUY_FLAGS}
    result["max_development_level"] = strategy["max_development_level"]
    result["unspendable_cash"] = strategy["unspendable_cash"]
    result["win"] = 1 if summary.get("winner") == HERO else 0
    result["props"] = hero_stats.get("props", 0)
    result["houses"] = hero_stats.get("houses", 0)
    result["hotels"] = hero_stats.get("hotels", 0)
    result["turns"] = hero_stats.get("turns", SimulationSettings.n_moves)
    result["game_seed"] = game_seed
    result["error"] = False

    # Add per-group fields (ever_*, partial_*, max_owned_*)
    for gn in GROUP_NAMES:
        k_ever = f"ever_{gn}"
        k_partial = f"partial_{gn}"
        k_max = f"max_owned_{gn}"
        if k_ever in hero_stats:
            result[k_ever] = hero_stats[k_ever]
        else:
            result[k_ever] = hero_stats.get(k_ever, 0)
        if k_partial in hero_stats:
            result[k_partial] = hero_stats[k_partial]
        else:
            result[k_partial] = hero_stats.get(k_partial, 0.0)
        if k_max in hero_stats:
            result[k_max] = hero_stats[k_max]
        else:
            result[k_max] = hero_stats.get(k_max, 0)

    return result


def main():
    LogSettings.init_logs()

    # CSV header: buy flags + params + outputs + per-group fields
    fieldnames = BUY_FLAGS + ["max_development_level", "unspendable_cash",
                              "win", "props", "houses", "hotels", "turns", "game_seed", "error"]
    for gn in GROUP_NAMES:
        fieldnames += [f"ever_{gn}", f"partial_{gn}", f"max_owned_{gn}"]

    # Build jobs
    rng = random.Random(SimulationSettings.seed)
    jobs = []
    game_counter = 0
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

    total_jobs = len(jobs)
    print(f"Prepared {total_jobs} jobs (strategies * games). Launching {N_WORKERS or 'default'} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_game, job): job for job in jobs}
        completed = 0
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": True, "error_message": str(e), "strategy": job[0], "game_seed": job[2]}
            results.append(res)
            completed += 1
            if not res.get("error"):
                print(f"Completed {completed}/{total_jobs} seed={res['game_seed']} win={res['win']}")
            else:
                print(f"Error in job {completed}/{total_jobs}: {res.get('error_message')}")

    # Sort: by buy flags then max_development_level then unspendable_cash then game_seed
    def sort_key(r):
        # error rows may not include keys
        return tuple((r.get(k, "") for k in BUY_FLAGS)) + (r.get("max_development_level", 0), r.get("unspendable_cash", 0), r.get("game_seed", 0))

    results_sorted = sorted(results, key=sort_key)

    with open("strategy_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results_sorted:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    print("Results saved to strategy_results.csv")


if __name__ == "__main__":
    main()
