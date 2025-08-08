# inspect_simulator.py
import inspect
import settings
from monopoly.core.game import monopoly_game

print("— GameSettings class —")
print(inspect.getsource(settings.GameSettings))

print("\n— monopoly_game signature —")
print(inspect.signature(monopoly_game))
