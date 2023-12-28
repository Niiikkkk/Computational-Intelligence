from game import Player,Game,Move
import numpy as np
import random
import math

class MinMaxPlayer(Player):
    def __init__(self, playerPlaying, levels_depth):
        self.moves_value = []
        self.playerPlaying = playerPlaying
        self.levels_depth = levels_depth

    def game_evaluation(self,game:Game,depth):
        """Playing for X, if it wins, the reward is 1 otherwise is -1, the oppisite when playing for O"""
        win = game.check_winner()
        ret = 0 + depth
        if win == 0:
            ret = 100+ depth
        elif win == 1:
            ret = -100-depth
        return ret
    
    def min_max(self, game:Game, alpha, beta, depth):
        if depth <= 0 or game.check_winner() != -1:
            return self.game_evaluation(game,depth), None
        best_move = None
        if game.current_player == 0:
            for move in game.get_possible_moves(game.current_player):
                tmp = game.get_board()
                g = Game()
                g.set_board(tmp)
                g.current_player = 0
                g.make_move(move[0],move[1])
                g.current_player = 1
                eval , _ = self.min_max(g,alpha,beta,depth-1)
                if eval > alpha:
                    alpha = eval
                    best_move = move
                if beta <= alpha:
                    break
            return alpha, best_move
        else:
            for move in game.get_possible_moves(game.current_player):
                tmp = game.get_board()
                g=Game()
                g.set_board(tmp)
                g.current_player = 1
                g.make_move(move[0],move[1])
                g.current_player = 0
                eval , _ = self.min_max(g,alpha,beta,depth-1)
                if eval < beta:
                    beta = eval
                    best_move = move
                if beta <= alpha:
                    break
            return beta, best_move
        
        ### DOPO UN PO' NON MI CAMBIA PIù LA BOARD!!!!
        
    def choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        possibleMoves = game.get_possible_moves(game.current_player)
        possibleMoveCount = len(possibleMoves)

        depth = 0
        for depthLvl in self.levels_depth:
            if (possibleMoveCount > depthLvl[1]):
                depth = depthLvl[0]
            else:
                break

        _, move = self.min_max(game, -math.inf, math.inf, depth)

        return move

    def give_rew(self, reward):
        pass