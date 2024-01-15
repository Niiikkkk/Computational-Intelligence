# Quixo
This project is a game of Quixo, a board game similar to Tic-Tac-Toe. The game is played on a 5x5 board, and the goal 
is to get five of your pieces in a row. The catch is that you can only move the pieces on the outside of the board, and you
can only move them in a straight line. The game is played by two players, and each player has five pieces. The game is 
played by moving a piece to an empty space, and then pushing the row or column of the piece in the direction of the move.

## Files

- `game.py`: This file is used to play the game using the different players.
- `CustomGameClass.py`: This file contains the custom class that extend ``game.py``.
- `Quixo.ipynb`: This notebook contains the players with their testing.
- `RL_player_1`: This file contains the policy of the first Reinforcement Learning player.
- `RL_player_2`: This file contains the policy of the second Reinforcement Learning player.

## Players

We develop two players:
- Reinforcement Learning player trained against a random player.
- Minimax player

We tried a lot of different parameters for the Reinforcement Learning player, but we saved the best two: ``RL_player_1``
and ``RL_player_2``. The parameters are reported in the notebook.

## Collaboration
For this project, I collaborate with [Nicholas Berardo s319441](https://github.com/Niiikkkk/Computational-Intelligence/tree/main)