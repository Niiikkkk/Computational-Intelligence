import random
from game import Game, Move, Player
from QL_Player import QLPlayer, train, test
from RL_Player import RLPlayer, train, test
from MinMax_Player import MinMaxPlayer

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def choose_action(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

    def give_rew(self, reward):
        pass

    def add_state(self, s):
        pass


if __name__ == '__main__':
    p = MinMaxPlayer()
    g = Game()

    val = p.min_max(g,0,1)