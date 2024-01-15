from game import Player, Game, Move

import numpy as np


class Quixo(Game):
    def __init__(self) -> None:
        super().__init__()

    def set_board(self, b):
        """
        Set the board
        """
        self._board = b

    def switch_player(self):
        """
        Switch the current player
        """
        if self.current_player == 1:
            self.current_player = 0
        else:
            self.current_player = 1

    def reset(self):
        """
        Reset the board
        """
        self._board = np.ones((5, 5), dtype=np.uint8) * -1

    def print(self):
        """Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1"""
        # define a board for pretty printing
        id_to_block = {-1: '⬜️', 0: '❌', 1: '⭕️'}
        fancy_board = np.chararray(self.get_board().shape, itemsize=1, unicode=True)
        for i in range(fancy_board.shape[0]):
            for j in range(fancy_board.shape[1]):
                # fill the fancy board
                fancy_board[(i, j)] = id_to_block[self.get_board()[(i, j)]]
        print(fancy_board)

    def play(self, player1: Player, player2: Player, print_flag: bool = False) -> int:
        """Play the game. Returns the winning player"""
        players = [player1, player2]
        self.current_player = 1
        winner = -1
        max_moves = 200
        while winner < 0 and max_moves > 0:
            self.current_player += 1
            self.current_player %= len(players)
            ok = False
            if print_flag:
                #print(f" Player {players[self.current_player].__class__.__name__} turn")
                self.print()
            while not ok:
                # from_pos is the position, for example [0,3]
                # slide is one element of Move (top,left...)
                from_pos, slide = players[self.current_player].make_move(self)
                ok = self.__move(from_pos, slide, self.current_player)
            max_moves -= 1
            #print(f"Player {players[self.current_player].__class__.__name__} moved {from_pos} {slide}")
            winner = self.check_winner()
        return winner

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        return super()._Game__move(from_pos, slide, player_id)

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        return super()._Game__take(from_pos, player_id)

    def slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        return super()._Game__slide(from_pos, slide)