from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum

import numpy as np


# Rules on PDF


class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def choose_action(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass

    @abstractmethod
    def give_rew(self, reward):
        pass


class Game(object):
    def __init__(self) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player = 0

    def get_board(self):
        """
        Returns the board
        """
        return deepcopy(self._board)

    def set_board(self, b):
        self._board = b

    def get_hash_board(self):
        return str(self._board.reshape(5 * 5))

    def convert_matrix_board_to_tuple(self, board):
        current_board = tuple(tuple(riga) for riga in board)
        return current_board

    def get_current_player(self) -> int:
        """
        Returns the current player
        """
        return deepcopy(self.current_player)

    def switch_player(self):
        if self.current_player == 1:
            self.current_player = 0
        else:
            self.current_player = 1

    def reset(self):
        self._board = np.ones((5, 5), dtype=np.uint8) * -1

    def get_possible_moves(self, player):
        # possible moves:
        # - take border empty and fill the hole by moving in the 3 directions
        # - take one of your blocks on the border and fill the hole by moving in the 3 directions
        # 44 at start possible moves
        pos = set()
        for r in [0, 4]:
            for c in range(5):
                if self._board[r, c] == -1 or self._board[r, c] == player:
                    if r == 0 and c == 0: #OK
                        pos.add(((c, r), Move.BOTTOM))
                        pos.add(((c, r), Move.RIGHT))
                    elif r == 0 and c == 4: #OK
                        pos.add(((c, r), Move.BOTTOM))
                        pos.add(((c, r), Move.LEFT))
                    elif r == 4 and c == 0: #OK
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.RIGHT))
                    elif r == 4 and c == 4: #OK
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.LEFT))
                    elif r == 0: #OK
                        pos.add(((c, r), Move.BOTTOM))
                        pos.add(((c, r), Move.LEFT))
                        pos.add(((c, r), Move.RIGHT))
                    elif r == 4: #OK
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.LEFT))
                        pos.add(((c, r), Move.RIGHT))
        for c in [0, 4]:
            for r in range(5):
                if self._board[r, c] == -1 or self._board[r, c] == player:
                    if r == 0 and c == 0: #OK
                        pos.add(((c, r), Move.BOTTOM))
                        pos.add(((c, r), Move.RIGHT))
                    elif r == 0 and c == 4: #OK
                        pos.add(((c, r), Move.BOTTOM))
                        pos.add(((c, r), Move.LEFT))
                    elif r == 4 and c == 0: #OK
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.RIGHT))
                    elif r == 4 and c == 4: #OK
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.LEFT))
                    elif c == 0: 
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.RIGHT))
                        pos.add(((c, r), Move.BOTTOM))
                    elif c == 4:
                        pos.add(((c, r), Move.TOP))
                        pos.add(((c, r), Move.LEFT))
                        pos.add(((c, r), Move.BOTTOM))
        return list(pos)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        print(self._board)

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return the relative id
                return self._board[x, 0]
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                return self._board[0, y]
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
                [self._board[x, x]
                 for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            return self._board[0, 0]
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
                [self._board[x, -(x + 1)]
                 for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            return self._board[0, -1]
        return -1

    def play(self, player1: Player, player2: Player) -> int:
        """Play the game. Returns the winning player"""
        players = [player1, player2]
        self.current_player = 1
        winner = -1
        while winner < 0:
            self.current_player += 1
            self.current_player %= len(players)
            ok = False
            while not ok:
                # from_pos is the position, for example [0,3]
                # slide is one element of Move (top,left...)
                from_pos, slide = players[self.current_player].choose_action(self)
                ok = self.make_move(from_pos, slide)
            winner = self.check_winner()
        return winner

    def make_move(self, from_pos: tuple[int, int], slide: Move) -> bool:
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.take((from_pos[1], from_pos[0]))
        if acceptable:
            acceptable = self.slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def take(self, from_pos: tuple[int, int]) -> bool:
        """Take piece"""
        # acceptable only if in border
        acceptable: bool = (
                               # check if it is in the first row
                                   (from_pos[0] == 0 and from_pos[1] < 5)
                                   # check if it is in the last row
                                   or (from_pos[0] == 4 and from_pos[1] < 5)
                                   # check if it is in the first column
                                   or (from_pos[1] == 0 and from_pos[0] < 5)
                                   # check if it is in the last column
                                   or (from_pos[1] == 4 and from_pos[0] < 5)
                               # and check if the piece can be moved by the current player
                           ) and (self._board[from_pos] < 0 or self._board[from_pos] == self.current_player)
        if acceptable:
            self._board[from_pos] = self.current_player
        return acceptable

    def slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        """Slide the other pieces"""
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                    slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                    slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                    slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                    slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                    slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                    slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
