""" Function that wraps one game of monopoly:
1. Setting up the board,
2. Players
3. Making moves by all players
"""
from typing import Tuple, Dict, Any

from monopoly.core.move_result import MoveResult
from monopoly.core.board import Board
from monopoly.core.dice import Dice
from monopoly.core.game_utils import assign_property, _check_end_conditions, log_players_and_board_state
from monopoly.core.player import Player
from monopoly.log import Log
from monopoly.log_settings import LogSettings
from settings import SimulationSettings, GameSettings, GameMechanics


def monopoly_game(game_number_and_seeds: Tuple[int,int]) -> Dict[str, Any]:
    """ Simulation of one game.
    Parameters packed into a tuple: (game_number, game_seed)
    """
    game_number, game_seed = game_number_and_seeds
    board, dice, events_log, bankruptcies_log = setup_game(game_number, game_seed)

    # Set up players with their behavior settings, starting money and properties.
    players = setup_players(board, dice)

    # Play the game until end conditions
    for turn_n in range(1, SimulationSettings.n_moves + 1):
        events_log.add(f"\n== GAME {game_number} Turn {turn_n} ===")
        log_players_and_board_state(board, events_log, players)
        board.log_board_state(events_log)
        events_log.add("")

        if _check_end_conditions(players, events_log, game_number, turn_n):
            break

        # Players make their moves
        for player in players:
            if player.is_bankrupt:
                continue
            move_result = player.make_a_move(board, players, dice, events_log)
            if move_result == MoveResult.BANKRUPT:
                bankruptcies_log.add(f"{game_number}\t{player}\t{turn_n}")

    # log the final game state
        board.log_current_map(events_log)
    events_log.save()
    if bankruptcies_log.content:
        bankruptcies_log.save()

    # —— Build and return a summary dict —— 

    survivors = [p for p in players if not p.is_bankrupt]
    if len(survivors) == 1:
        winner = survivors[0].name
    else:
        winner = max(survivors, key=lambda p: p.money).name

    summary_players = {}

    # helper to normalize group name to string key
    def normalize_group_name(g):
        return str(g).lower().replace(" ", "_").replace("'", "").replace("-", "_")

    groups = list(board.groups.keys())

    for p in players:
        # Determine initial cash
        start_money = GameSettings.starting_money
        if isinstance(start_money, dict):
            initial_cash = start_money.get(p.name, 0)
        else:
            initial_cash = start_money

        # Compute ROI (optional, kept for compatibility; can be removed if you prefer)
        final_net = p.net_worth()
        roi = final_net / initial_cash if initial_cash > 0 else None

        # base stats
        player_summary = {
            # note: you asked to ignore roi as inaccurate for ML — include if you still want it
            "roi": roi,
            "win": winner == p.name,
            "props": len(p.owned),
            "houses": sum(c.has_houses for c in p.owned),
            "hotels": sum(c.has_hotel for c in p.owned),
            "turns": turn_n
        }

        # per-group features
        for group in groups:
            norm = normalize_group_name(group)
            group_cells = board.groups[group]
            total_in_group = len(group_cells)
            final_owned = sum(1 for c in group_cells if c.owner == p)
            partial_frac = final_owned / total_in_group if total_in_group > 0 else 0.0
            ever_owned = 1 if group in getattr(p, "ever_owned_sets", set()) else 0
            max_owned = int(getattr(p, "max_owned_in_group", {}).get(group, 0))

            player_summary[f"ever_{norm}"] = ever_owned
            player_summary[f"partial_{norm}"] = partial_frac
            player_summary[f"max_owned_{norm}"] = max_owned

        summary_players[p.name] = player_summary

    return {
        "winner": winner,
        "players": summary_players
    }


def setup_players(board, dice):
    players = [Player(player_name, player_setting)
               for player_name, player_setting in GameSettings.players_list]

    if GameSettings.shuffle_players:
        dice.shuffle(players)

    # Set up players starting money according to the game settings:
    starting_money = GameSettings.starting_money
    if isinstance(starting_money, dict):
        for player in players:
            player.money = starting_money.get(player.name, 0)
    else:
        for player in players:
            player.money = starting_money

    # set up players' initial properties
    for player in players:
        property_indices = GameSettings.starting_properties.get(player.name, [])
        for cell_index in property_indices:
            assign_property(player, board.cells[cell_index], board)
        # Update ownership history after assigning starting properties
        try:
            player._update_group_counts(board)
        except Exception:
            pass

    return players


def setup_game(game_number, game_seed):
    events_log = Log(LogSettings.EVENTS_LOG_PATH, disabled=not LogSettings.KEEP_GAME_LOG)
    events_log.add(f"= GAME {game_number} of {SimulationSettings.n_games} (seed = {game_seed}) =")

    bankruptcies_log = Log(LogSettings.BANKRUPTCIES_PATH)

    # Initialize the board (plots, chance, community chest etc.)
    board = Board(GameSettings)
    dice = Dice(game_seed, GameMechanics.dice_count, GameMechanics.dice_sides, events_log)
    dice.shuffle(board.chance.cards)
    dice.shuffle(board.chest.cards)
    return board, dice, events_log, bankruptcies_log
