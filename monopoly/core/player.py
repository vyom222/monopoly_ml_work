""" Player Class
"""
from collections import defaultdict
from monopoly.core.cell import (
    GoToJail, LuxuryTax, IncomeTax, FreeParking,
    Chance, CommunityChest, Property
)
from monopoly.core.constants import INDIGO, BROWN, RAILROADS, UTILITIES
from monopoly.core.move_result import MoveResult
from settings import GameMechanics


class Player:
    """ Class to contain player-related info and actions:
    - money, position, owned property
    - actions to buy property or handle Chance cards etc.
    """

    def __init__(self, name, settings):

        # Player's name and behavioral settings
        self.name = name
        self.settings = settings

        # Player's money (will be set up by the simulation)
        self.money = 0

        # Player's position
        self.position = 0

        # Person's roll double and jail status
        self.in_jail = False
        self.had_doubles = 0
        self.days_in_jail = 0
        self.get_out_of_jail_chance = False
        self.get_out_of_jail_comm_chest = False

        # Owned properties
        self.owned = []

        # List of properties the player wants to sell / buy
        # through trading with other players
        self.wants_to_sell = set()
        self.wants_to_buy = set()

        # Bankrupt (game ended for this player)
        self.is_bankrupt = False

        # Placeholder for various flags used throughout the game
        self.other_notes = ""

        # --- New tracking for simulations / ML ---
        # Set of groups (color constants) that the player has ever had a full monopoly for
        self.ever_owned_sets = set()
        # Max number of properties the player has ever held in each group at any point in time
        # e.g. {GROUP_CONST: 2, ...}
        self.max_owned_in_group = defaultdict(int)

    def __str__(self):
        return self.name

    # -------------------- helpers for tracking ownership history --------------------
    def _update_group_counts(self, board):
        """
        Update `max_owned_in_group` and `ever_owned_sets` according to current ownership.
        Call whenever ownership changes (buy, trade, bankruptcy transfer, starting ownership).
        """
        for group, cells in board.groups.items():
            owned_count = sum(1 for c in cells if c.owner == self)
            # update max seen
            if owned_count > self.max_owned_in_group.get(group, 0):
                self.max_owned_in_group[group] = owned_count
            # full monopoly -> mark ever_owned_sets
            if owned_count == len(cells) and len(cells) > 0:
                self.ever_owned_sets.add(group)

    # -------------------- net worth --------------------
    def net_worth(self, count_mortgaged_as_full_value=False):
        """ Calculate player's net worth (cash + property + houses)
        count_mortgaged_as_full_value determines if we consider property mortgaged status:
        - True: count as full, for Income Tax calculation
        - False: count partially, for net worth statistics
        """
        net_worth = int(self.money)

        for cell in self.owned:

            if cell.is_mortgaged and not count_mortgaged_as_full_value:
                # Partially count mortgaged properties
                net_worth += int(cell.cost_base * (1 - GameMechanics.mortgage_value))
            else:
                net_worth += cell.cost_base
                net_worth += (cell.has_houses + cell.has_hotel) * cell.cost_house

        return net_worth

    # -------------------- main turn --------------------
    def make_a_move(self, board, players, dice, log) -> MoveResult:
        """ Main function for a player to make a move
        Returns:
            MoveResult: CONTINUE, BANKRUPT, END_MOVE
        """

        # If the player bankrupt - do nothing
        if self.is_bankrupt:
            return MoveResult.BANKRUPT

        log.add(f"=== {self.name} (${self.money}, " +
                f"at {board.cells[self.position].name}) goes: ===")

        # Before rolling the dice:
        while self.do_a_two_way_trade(players, board, log):
            pass
        while self.unmortgage_a_property(board, log):
            pass
        self.improve_properties(board, log)

        # The move itself: roll dice
        _, dice_sum, is_double = dice.roll()

        # Third double -> jail
        if is_double and self.had_doubles == 2:
            self.handle_going_to_jail("rolled 3 doubles in a row", log)
            return MoveResult.END_MOVE

        # In jail handling
        if self.in_jail:
            if self.is_player_stay_in_jail(is_double, board, log):
                return MoveResult.END_MOVE

        # Move
        self.position += dice_sum
        if self.position >= 40:
            self.handle_salary(board, log)
        self.position %= 40
        log.add(f"{self.name} goes to: {board.cells[self.position].name}")

        # Special cells
        if isinstance(board.cells[self.position], Chance):
            if self.handle_chance(board, players, log) == MoveResult.END_MOVE:
                return MoveResult.END_MOVE

        if isinstance(board.cells[self.position], CommunityChest):
            if self.handle_community_chest(board, players, log) == MoveResult.END_MOVE:
                return MoveResult.END_MOVE

        if isinstance(board.cells[self.position], Property):
            self.handle_landing_on_property(board, players, dice, log)

        if isinstance(board.cells[self.position], GoToJail):
            self.handle_going_to_jail("landed on Go To Jail", log)
            return MoveResult.END_MOVE

        if isinstance(board.cells[self.position], FreeParking):
            if GameMechanics.free_parking_money:
                log.add(f"{self} gets ${board.free_parking_money} from Free Parking")
                self.money += board.free_parking_money
                board.free_parking_money = 0

        if isinstance(board.cells[self.position], LuxuryTax):
            self.pay_money(GameMechanics.luxury_tax, "bank", board, log)
            if not self.is_bankrupt:
                log.add(f"{self} pays Luxury Tax ${GameMechanics.luxury_tax}")

        if isinstance(board.cells[self.position], IncomeTax):
            self.handle_income_tax(board, log)

        self.other_notes = ""

        if self.is_bankrupt:
            return MoveResult.BANKRUPT

        if is_double:
            self.had_doubles += 1
            log.add(f"{self} rolled a double ({self.had_doubles} in a row) so they go again.")
            return self.make_a_move(board, players, dice, log)

        self.had_doubles = 0
        return MoveResult.END_MOVE

    # -------------------- helpers for jail / salary --------------------
    def handle_salary(self, board, log):
        """ Adding Salary to the player's money """
        self.money += board.settings.mechanics.salary
        log.add(f" {self.name} receives salary ${board.settings.mechanics.salary}")

    def handle_going_to_jail(self, message, log):
        """ Start the jail time """
        log.add(f"{self} {message}, and goes to Jail.")
        self.position = 10
        self.in_jail = True
        self.had_doubles = 0
        self.days_in_jail = 0

    def is_player_stay_in_jail(self, dice_roll_is_double, board, log):
        """ Handle a player being in Jail """
        if self.get_out_of_jail_chance or self.get_out_of_jail_comm_chest:
            log.add(f"{self} uses a GOOJF card")
            self.in_jail = False
            self.days_in_jail = 0
            if self.get_out_of_jail_chance:
                board.chance.add("Get Out of Jail Free")
                self.get_out_of_jail_chance = False
            else:
                board.chest.add("Get Out of Jail Free")
                self.get_out_of_jail_comm_chest = False
        elif dice_roll_is_double:
            log.add(f"{self} rolled a double, a leaves jail for free")
            self.in_jail = False
            self.days_in_jail = 0
        elif self.days_in_jail == 2:
            log.add(f"{self} did not rolled a double for the third time, " +
                    f"pays {GameMechanics.exit_jail_fine} and leaves jail")
            self.pay_money(GameMechanics.exit_jail_fine, "bank", board, log)
            self.in_jail = False
            self.days_in_jail = 0
        else:
            log.add(f"{self} stays in jail")
            self.days_in_jail += 1
            return True
        return False

    # -------------------- chance / community chest --------------------
    def handle_chance(self, board, players, log):
        card = board.chance.draw()
        log.add(f"{self} drew Chance card: '{card}'")

        if card == "Advance to Boardwalk":
            log.add(f"{self} goes to {board.cells[39]}")
            self.position = 39

        elif card == "Advance to Go (Collect $200)":
            log.add(f"{self} goes to {board.cells[0]}")
            self.position = 0
            self.handle_salary(board, log)

        elif card == "Advance to Illinois Avenue. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[24]}")
            if self.position > 24:
                self.handle_salary(board, log)
            self.position = 24

        elif card == "Advance to St. Charles Place. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[11]}")
            if self.position > 11:
                self.handle_salary(board, log)
            self.position = 11

        elif card == "Take a trip to Reading Railroad. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[5]}")
            if self.position > 5:
                self.handle_salary(board, log)
            self.position = 5

        elif card == "Go Back 3 Spaces":
            self.position -= 3
            log.add(f"{self} goes to {board.cells[self.position]}")

        elif card == "Advance to the nearest Railroad. " + \
                "If owned, pay owner twice the rental to which they are otherwise entitled":
            nearest_railroad = self.position
            while (nearest_railroad - 5) % 10 != 0:
                nearest_railroad += 1
                nearest_railroad %= 40
            log.add(f"{self} goes to {board.cells[nearest_railroad]}")
            if self.position > nearest_railroad:
                self.handle_salary(board, log)
            self.position = nearest_railroad
            self.other_notes = "double rent"

        elif card == "Advance token to nearest Utility. " + \
                "If owned, throw dice and pay owner a total ten times amount thrown.":
            nearest_utility = self.position
            while nearest_utility not in (12, 28):
                nearest_utility += 1
                nearest_utility %= 40
            log.add(f"{self} goes to {board.cells[nearest_utility]}")
            if self.position > nearest_utility:
                self.handle_salary(board, log)
            self.position = nearest_utility
            self.other_notes = "10 times dice"

        elif card == "Get Out of Jail Free":
            log.add(f"{self} now has a 'Get Out of Jail Free' card")
            self.get_out_of_jail_chance = True
            board.chance.remove("Get Out of Jail Free")

        elif card == "Go to Jail. Go directly to Jail, do not pass Go, do not collect $200":
            self.handle_going_to_jail("got GTJ Chance card", log)
            return MoveResult.END_MOVE

        elif card == "Bank pays you dividend of $50":
            log.add(f"{self} gets $50")
            self.money += 50

        elif card == "Your building loan matures. Collect $150":
            log.add(f"{self} gets $150")
            self.money += 150

        elif card == "Speeding fine $15":
            self.pay_money(15, "bank", board, log)

        elif card == "Make general repairs on all your property. For each house pay $25. " + \
                "For each hotel pay $100":
            repair_cost = sum(cell.has_houses * 25 + cell.has_hotel * 100 for cell in self.owned)
            log.add(f"Repair cost: ${repair_cost}")
            self.pay_money(repair_cost, "bank", board, log)

        elif card == "You have been elected Chairman of the Board. Pay each player $50":
            for other_player in players:
                if other_player != self and not other_player.is_bankrupt:
                    self.pay_money(50, other_player, board, log)
                    if not self.is_bankrupt:
                        log.add(f"{self} pays {other_player} $50")

        return ""

    def handle_community_chest(self, board, players, log):
        card = board.chest.draw()
        log.add(f"{self} drew Community Chest card: '{card}'")

        if card == "Advance to Go (Collect $200)":
            log.add(f"{self} goes to {board.cells[0]}")
            self.position = 0
            self.handle_salary(board, log)

        elif card == "Get Out of Jail Free":
            log.add(f"{self} now has a 'Get Out of Jail Free' card")
            self.get_out_of_jail_comm_chest = True
            board.chest.remove("Get Out of Jail Free")

        elif card == "Go to Jail. Go directly to Jail, do not pass Go, do not collect $200":
            self.handle_going_to_jail("got GTJ Community Chest card", log)
            return MoveResult.END_MOVE

        elif card == "Doctor's fee. Pay $50":
            self.pay_money(50, "bank", board, log)

        elif card == "Pay hospital fees of $100":
            self.pay_money(100, "bank", board, log)

        elif card == "Pay school fees of $50":
            self.pay_money(50, "bank", board, log)

        elif card == "You are assessed for street repair. $40 per house. $115 per hotel":
            repair_cost = sum(cell.has_houses * 40 + cell.has_hotel * 115 for cell in self.owned)
            log.add(f"Repair cost: ${repair_cost}")
            self.pay_money(repair_cost, "bank", board, log)

        elif card == "Bank error in your favor. Collect $200":
            log.add(f"{self} gets $200")
            self.money += 200

        elif card == "From sale of stock you get $50":
            log.add(f"{self} gets $50")
            self.money += 50

        elif card == "Holiday fund matures. Receive $100":
            log.add(f"{self} gets $100")
            self.money += 100

        elif card == "Income tax refund. Collect $20":
            log.add(f"{self} gets $20")
            self.money += 20

        elif card == "Life insurance matures. Collect $100":
            log.add(f"{self} gets $100")
            self.money += 100

        elif card == "Receive $25 consultancy fee":
            log.add(f"{self} gets $25")
            self.money += 25

        elif card == "You have won second prize in a beauty contest. Collect $10":
            log.add(f"{self} gets $10")
            self.money += 10

        elif card == "You inherit $100""You inherit $100":
            log.add(f"{self} gets $100")
            self.money += 100

        elif card == "It is your birthday. Collect $10 from every player":
            for other_player in players:
                if other_player != self and not other_player.is_bankrupt:
                    other_player.pay_money(50, self, board, log)
                    if not other_player.is_bankrupt:
                        log.add(f"{other_player} pays {self} $10")

        return ""

    # -------------------- taxes / landing on property --------------------
    def handle_income_tax(self, board, log):
        tax_to_pay = min(
            GameMechanics.income_tax,
            int(GameMechanics.income_tax_percentage *
                self.net_worth(count_mortgaged_as_full_value=True)))

        if tax_to_pay == GameMechanics.income_tax:
            log.add(f"{self} pays fixed Income tax {GameMechanics.income_tax}")
        else:
            log.add(f"{self} pays {GameMechanics.income_tax_percentage * 100:.0f}% " +
                    f"Income tax {tax_to_pay}")
        self.pay_money(tax_to_pay, "bank", board, log)

    def handle_landing_on_property(self, board, players, dice, log):
        """ Landing on property: either buy it or pay rent """

        def is_willing_to_buy_property(property_to_buy):
            """ Check if the player is willing to buy an unowned property """
            if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
                return False
            if property_to_buy.cost_base > self.money:
                return False
            if property_to_buy.group in self.settings.ignore_property_groups:
                return False
            return True

        def buy_property(property_to_buy):
            """ Player buys the property """
            property_to_buy.owner = self
            self.owned.append(property_to_buy)
            self.money -= property_to_buy.cost_base
            # update ownership history
            try:
                self._update_group_counts(board)
            except Exception:
                pass

        # This is the property a player landed on
        landed_property = board.cells[self.position]

        # Property is not owned by anyone
        if landed_property.owner is None:
            if is_willing_to_buy_property(landed_property):
                buy_property(landed_property)
                log.add(f"{self.name} bought {landed_property} " +
                        f"for ${landed_property.cost_base}")

                board.recalculate_monopoly_multipliers(landed_property)
                for player in players:
                    player.update_lists_of_properties_to_trade(board)
            else:
                log.add(f"{self.name} landed on a {landed_property}, he refuses to buy it")
                # TODO: bank auctions property
        else:
            # It is mortgaged: no action
            if landed_property.is_mortgaged:
                log.add("Property is mortgaged, no rent")
            elif landed_property.owner == self:
                log.add("Own property, no rent")
            else:
                log.add(f"{self.name} landed on a property, " +
                        f"owned by {landed_property.owner}")
                rent_amount = landed_property.calculate_rent(dice)
                if self.other_notes == "double rent":
                    rent_amount *= 2
                    log.add(f"Per Chance card, rent is doubled (${rent_amount}).")
                if self.other_notes == "10 times dice":
                    rent_amount = rent_amount // landed_property.monopoly_multiplier * 10
                    log.add(f"Per Chance card, rent is 10x dice throw (${rent_amount}).")
                self.pay_money(rent_amount, landed_property.owner, board, log)
                if not self.is_bankrupt:
                    log.add(f"{self} pays {landed_property.owner} rent ${rent_amount}")

    # -------------------- improving / unmortgaging --------------------
    def improve_properties(self, board, log):
        def get_next_property_to_improve():
            can_be_improved = []
            for cell in self.owned:
                if (
                        cell.has_hotel == 0
                        and not cell.is_mortgaged
                        and cell.monopoly_multiplier == 2
                        and cell.group not in (RAILROADS, UTILITIES)
                ):
                    for other_cell in board.groups[cell.group]:
                        if (other_cell.has_houses < cell.has_houses and not other_cell.has_hotel) or other_cell.is_mortgaged:
                            break
                    else:
                        if cell.has_houses != 4 and board.available_houses > 0 or \
                                cell.has_houses == 4 and board.available_hotels > 0:
                            can_be_improved.append(cell)
            can_be_improved.sort(key=lambda x: x.cost_house)
            return can_be_improved[0] if can_be_improved else None

        while True:
            cell_to_improve = get_next_property_to_improve()
            if cell_to_improve is None:
                break

            log.add(f"Trying to build: {cell_to_improve}, houses={cell_to_improve.has_houses}, hotel={cell_to_improve.has_hotel}, max_dev={self.settings.max_development_level}")
            improvement_cost = cell_to_improve.cost_house

            if self.money - improvement_cost < self.settings.unspendable_cash:
                break

            ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}

            if cell_to_improve.has_houses != 4 and cell_to_improve.has_houses < self.settings.max_development_level:
                cell_to_improve.has_houses += 1
                board.available_houses -= 1
                self.money -= cell_to_improve.cost_house
                log.add(f"{self} built {ordinal[cell_to_improve.has_houses]} " +
                        f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")
                log.add(f"After build: {cell_to_improve} now has {cell_to_improve.has_houses} houses")
            elif cell_to_improve.has_houses == 4 and self.settings.max_development_level == 5:
                cell_to_improve.has_houses = 0
                cell_to_improve.has_hotel = 1
                board.available_houses += 4
                board.available_hotels -= 1
                self.money -= cell_to_improve.cost_house
                log.add(f"{self} built a hotel on {cell_to_improve}")
            else:
                break

    def unmortgage_a_property(self, board, log):
        for cell in self.owned:
            if cell.is_mortgaged:
                cost_to_unmortgage = \
                    cell.cost_base * GameMechanics.mortgage_value + \
                    cell.cost_base * GameMechanics.mortgage_fee
                if self.money - cost_to_unmortgage >= self.settings.unspendable_cash:
                    log.add(f"{self} unmortgages {cell} for ${cost_to_unmortgage}")
                    self.money -= cost_to_unmortgage
                    cell.is_mortgaged = False
                    self.update_lists_of_properties_to_trade(board)
                    return True
        return False

    # -------------------- raising money & mortgage --------------------
    def raise_money(self, required_amount, board, log):
        def get_next_property_to_downgrade(required_amount):
            can_be_downgrade = []
            can_be_downgrade_has_houses = False
            for cell in self.owned:
                if cell.has_houses > 0 or cell.has_hotel > 0:
                    for other_cell in board.groups[cell.group]:
                        if cell.has_hotel == 0 and (
                                other_cell.has_houses > cell.has_houses or other_cell.has_hotel > 0):
                            break
                    else:
                        can_be_downgrade.append(cell)
                        if cell.has_houses > 0:
                            can_be_downgrade_has_houses = True

            if len(can_be_downgrade) == 0:
                return None

            if can_be_downgrade_has_houses:
                can_be_downgrade = [x for x in can_be_downgrade if x.has_hotel == 0]

            can_be_downgrade.sort(key=lambda x: x.cost_house // 2)
            while True:
                if len(can_be_downgrade) == 1:
                    return can_be_downgrade[0]
                if can_be_downgrade[-2].cost_house // 2 < required_amount:
                    return can_be_downgrade[-1]
                can_be_downgrade.pop()

        def get_list_of_properties_to_mortgage():
            list_to_mortgage = []
            for cell in self.owned:
                if not cell.is_mortgaged:
                    list_to_mortgage.append(
                        (int(cell.cost_base * GameMechanics.mortgage_value), cell))
            list_to_mortgage.sort(key=lambda x: -x[0])
            return list_to_mortgage

        while True:
            money_to_raise = required_amount - self.money
            cell_to_deimprove = get_next_property_to_downgrade(money_to_raise)

            if cell_to_deimprove is None or money_to_raise <= 0:
                break

            sell_price = cell_to_deimprove.cost_house // 2
            if cell_to_deimprove.has_hotel:
                if board.available_houses >= 4:
                    cell_to_deimprove.has_hotel = 0
                    cell_to_deimprove.has_houses = 4
                    board.available_hotels += 1
                    board.available_houses -= 4
                    log.add(f"{self} sells a hotel on {cell_to_deimprove}, raising ${sell_price}")
                    self.money += sell_price
                else:
                    cell_to_deimprove.has_hotel = 0
                    cell_to_deimprove.has_houses = 0
                    board.available_hotels += 1
                    log.add(f"{self} sells a hotel and all houses on {cell_to_deimprove}, " +
                            f"raising ${sell_price * 5}")
                    self.money += sell_price * 5
            else:
                cell_to_deimprove.has_houses -= 1
                board.available_houses += 1
                ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
                log.add(f"{self} sells {ordinal[cell_to_deimprove.has_houses + 1]} " +
                        f"house on {cell_to_deimprove}, raising ${sell_price}")
                self.money += sell_price

        # Mortgage properties
        list_to_mortgage = get_list_of_properties_to_mortgage()
        while list_to_mortgage and self.money < required_amount:
            mortgage_price, cell_to_mortgage = list_to_mortgage.pop()
            cell_to_mortgage.is_mortgaged = True
            self.money += mortgage_price
            log.add(f"{self} mortgages {cell_to_mortgage}, raising ${mortgage_price}")

    # -------------------- payments & bankruptcy --------------------
    def pay_money(self, amount, payee, board, log):
        """ Function to pay money to another player (or bank). Triggers bankruptcy if necessary. """

        def count_max_raisable_money():
            max_raisable = self.money
            for cell in self.owned:
                if cell.has_houses > 0:
                    max_raisable += cell.cost_house * cell.has_houses // 2
                if cell.has_hotel > 0:
                    max_raisable += cell.cost_house * 5 // 2
                if not cell.is_mortgaged:
                    max_raisable += int(cell.cost_base * GameMechanics.mortgage_value)
            return max_raisable

        def transfer_all_properties(payee_obj, board_obj, log_obj):
            """Transfer all properties (used during bankruptcy). Uses the helper updates."""
            while self.owned:
                cell_to_transfer = self.owned.pop()

                # Transfer to a player
                if isinstance(payee_obj, Player):
                    cell_to_transfer.owner = payee_obj
                    payee_obj.owned.append(cell_to_transfer)
                    # Update both players' histories
                    try:
                        payee_obj._update_group_counts(board_obj)
                    except Exception:
                        pass
                else:
                    # Transfer to bank (owner becomes None)
                    cell_to_transfer.owner = None
                    cell_to_transfer.is_mortgaged = False

                board_obj.recalculate_monopoly_multipliers(cell_to_transfer)
                log_obj.add(f"{self} transfers {cell_to_transfer} to {payee_obj}")

            # After transfer, update this player's groups (they now own none)
            try:
                self._update_group_counts(board_obj)
            except Exception:
                pass

        # Regular transaction (sufficient cash)
        if amount < self.money:
            self.money -= amount
            if payee != "bank":
                payee.money += amount
            elif payee == "bank" and GameMechanics.free_parking_money:
                board.free_parking_money += amount
            return

        max_raisable_money = count_max_raisable_money()
        # Can pay but need to sell some things first
        if amount < max_raisable_money:
            log.add(f"{self} has ${self.money}, he can pay ${amount}, " +
                    "but needs to mortgage/sell some things for that")
            self.raise_money(amount, board, log)
            self.money -= amount
            if payee != "bank":
                payee.money += amount
            elif payee == "bank" and GameMechanics.free_parking_money:
                board.free_parking_money += amount
            return

        # Bankruptcy (can't pay even after selling and mortgaging all)
        else:
            log.add(f"{self} has to pay ${amount}, max they can raise is ${max_raisable_money}")
            self.is_bankrupt = True
            log.add(f"{self} is bankrupt")

            # Raise as much cash as possible to give payee
            self.raise_money(amount, board, log)
            log.add(f"{self} gave {payee} all their remaining money (${self.money})")
            if payee != "bank":
                payee.money += self.money
            elif payee == "bank" and GameMechanics.free_parking_money:
                board.free_parking_money += amount

            self.money = 0

            # Transfer all property (mortgaged at this point) to payee
            transfer_all_properties(payee, board, log)

            # Reset all trade settings
            self.wants_to_sell = set()
            self.wants_to_buy = set()

    # -------------------- trade helpers --------------------
    def update_lists_of_properties_to_trade(self, board):
        """ Update list of properties player is willing to sell / buy """
        if not getattr(self.settings, "is_willing_to_make_trades", False):
            return

        self.wants_to_sell = set()
        self.wants_to_buy = set()

        for group_cells in board.groups.values():
            owned_by_me = []
            owned_by_others = []
            not_owned = []
            for cell in group_cells:
                if cell.owner == self:
                    owned_by_me.append(cell)
                elif cell.owner is None:
                    not_owned.append(cell)
                else:
                    owned_by_others.append(cell)

            if not_owned:
                continue
            if len(owned_by_me) == 1:
                self.wants_to_sell.add(owned_by_me[0])
            if len(owned_by_others) == 1:
                self.wants_to_buy.add(owned_by_others[0])

    def do_a_two_way_trade(self, players, board, log):
        """ Look for and perform a two-way trade """
        def get_price_difference(gives, receives):
            cost_gives = sum(cell.cost_base for cell in gives) if gives else 0
            cost_receives = sum(cell.cost_base for cell in receives) if receives else 0
            diff_abs = cost_gives - cost_receives

            diff_giver, diff_receiver = float("inf"), float("inf")
            if receives and cost_receives > 0:
                diff_giver = cost_gives / cost_receives
            if gives and cost_gives > 0:
                diff_receiver = cost_receives / cost_gives

            return diff_abs, diff_giver, diff_receiver

        def remove_by_color(cells, color):
            return [cell for cell in cells if cell.group != color]

        def would_complete_set(player, property_cell):
            color_group = property_cell.group
            owned_in_group = [p for p in player.owned if p.group == color_group]
            total_in_group = len(board.groups[color_group])
            return (len(owned_in_group) + 1) == total_in_group

        def fair_deal(player_gives, player_receives, other_player):
            color_receives = [cell.group for cell in player_receives]
            color_gives = [cell.group for cell in player_gives]

            both_colors = set(color_receives + color_gives)
            if both_colors.issubset({UTILITIES, INDIGO, BROWN}):
                return [], []

            for questionable_color in [UTILITIES, INDIGO, BROWN]:
                if questionable_color in color_receives and questionable_color in color_gives:
                    if len(player_receives) > len(player_gives):
                        player_receives = remove_by_color(player_receives, questionable_color)
                    elif len(player_receives) < len(player_gives):
                        player_gives = remove_by_color(player_gives, questionable_color)
                    else:
                        player_receives = remove_by_color(player_receives, questionable_color)
                        player_gives = remove_by_color(player_gives, questionable_color)

            player_receives.sort(key=lambda x: -x.cost_base)
            player_gives.sort(key=lambda x: -x.cost_base)

            while player_gives and player_receives:
                diff_abs, diff_giver, diff_receiver = get_price_difference(player_gives, player_receives)
                completion_bonus_self = getattr(self.settings, "set_completion_trade_bonus", 0)
                completion_bonus_other = getattr(other_player.settings, "set_completion_trade_bonus", 0)

                if any(would_complete_set(self, prop) for prop in player_receives):
                    adjusted_diff_abs = diff_abs - completion_bonus_self
                else:
                    adjusted_diff_abs = diff_abs

                if any(would_complete_set(other_player, prop) for prop in player_gives):
                    adjusted_diff_abs = adjusted_diff_abs + completion_bonus_other

                if adjusted_diff_abs > self.settings.trade_max_diff_absolute or (diff_giver != float("inf") and diff_giver > self.settings.trade_max_diff_relative):
                    player_gives.pop()
                    continue

                if -adjusted_diff_abs > other_player.settings.trade_max_diff_absolute or (diff_receiver != float("inf") and diff_receiver > other_player.settings.trade_max_diff_relative):
                    player_receives.pop()
                    continue

                break

            return player_gives, player_receives

        for other_player in players:
            if other_player is self:
                continue

            if self.wants_to_buy.intersection(other_player.wants_to_sell) and \
                    self.wants_to_sell.intersection(other_player.wants_to_buy):
                player_receives = list(self.wants_to_buy.intersection(other_player.wants_to_sell))
                player_gives = list(self.wants_to_sell.intersection(other_player.wants_to_buy))

                player_gives, player_receives = fair_deal(player_gives, player_receives, other_player)

                if player_receives and player_gives:
                    price_difference, _, _ = get_price_difference(player_gives, player_receives)

                    completion_bonus_self = getattr(self.settings, "set_completion_trade_bonus", 0)
                    completion_bonus_other = getattr(other_player.settings, "set_completion_trade_bonus", 0)

                    money_transfer = price_difference - completion_bonus_self + completion_bonus_other

                    # Validate payment ability
                    if money_transfer > 0:
                        if other_player.money - money_transfer < other_player.settings.unspendable_cash:
                            continue
                        other_player.money -= money_transfer
                        self.money += money_transfer
                    elif money_transfer < 0:
                        if self.money - abs(money_transfer) < self.settings.unspendable_cash:
                            continue
                        other_player.money += abs(money_transfer)
                        self.money -= abs(money_transfer)

                    # Property changes hands
                    for cell_to_receive in player_receives:
                        cell_to_receive.owner = self
                        self.owned.append(cell_to_receive)
                        other_player.owned.remove(cell_to_receive)
                    for cell_to_give in player_gives:
                        cell_to_give.owner = other_player
                        other_player.owned.append(cell_to_give)
                        self.owned.remove(cell_to_give)

                    # Update ownership history for both players
                    try:
                        self._update_group_counts(board)
                        other_player._update_group_counts(board)
                    except Exception:
                        pass

                    # Log the trade and compensation payment
                    log.add(f"Trade: {self} gives {[str(cell) for cell in player_gives]}, " +
                            f"receives {[str(cell) for cell in player_receives]} " +
                            f"from {other_player}")

                    if money_transfer > 0:
                        log.add(f"{self} received price difference compensation ${abs(money_transfer)} from {other_player}")
                    if money_transfer < 0:
                        log.add(f"{other_player} received price difference compensation ${abs(money_transfer)} from {self}")

                    # Recalculate monopoly and improvement status
                    if player_gives:
                        board.recalculate_monopoly_multipliers(player_gives[0])
                    if player_receives:
                        board.recalculate_monopoly_multipliers(player_receives[0])

                    # Recalculate who wants to buy what
                    for player in players:
                        player.update_lists_of_properties_to_trade(board)

                    # Return True to run a trading function again
                    return True

        return False
