import pickle
import random
from game import Game, Move, Player
from tqdm.auto import tqdm


class RLPlayer(Player):
    def __init__(self, player, alpha, epsilon, discount_factor) -> None:
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.player = player
        self.states = []
        self.state_value = {}

    def add_state(self, state):
        self.states.append(state)

    def get_state_value(self, state):
        if state not in self.state_value:
            self.state_value[state] = 0.0
        return self.state_value[state]

    def reset(self):
        self.states = []

    def choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        available_moves = game.get_possible_moves(self.player)
        if random.uniform(0, 1) < self.epsilon:  # do exploration for 30% of the time
            return random.choice(available_moves)
        else:  # take the best one for 70% of the time
            value_max = -999
            for move in available_moves:
                tmp = game.get_board()
                game.make_move(move[0], move[1])
                next_status = game.convert_matrix_board_to_tuple(game.get_board())
                game.set_board(tmp)
                value = 0 if self.state_value.get(next_status) is None else self.state_value.get(next_status)
                if value > value_max:
                    value_max = value
                    action = move
        return action

    def give_rew(self, reward):
        for st in reversed(self.states):
            if self.state_value.get(st) is None:
                self.state_value[st] = 0
            current_q_value = self.state_value[st]
            reward = current_q_value + self.alpha * (self.discount_factor * reward - current_q_value)
            self.state_value[st] = reward

    def save_policy(self, name):
        fw = open(name, 'wb')
        pickle.dump(self.state_value, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self.state_value = pickle.load(fr)
        fr.close()


def train(p1: RLPlayer, p2: RLPlayer, game, epochs):
    for epoch in tqdm(range(epochs)):
        game.reset()
        p1.reset()
        p2.reset()
        while game.check_winner() == -1:
            # Player 1
            ok = False
            while not ok:
                from_pos, slide = p1.choose_action(game)
                ok = game.make_move(from_pos, slide)
            p1.add_state(game.convert_matrix_board_to_tuple(game.get_board()))

            win = game.check_winner()
            if win != -1:
                if win == 0:
                    p1.give_rew(1)
                    p2.give_rew(0)
                else:
                    p1.give_rew(0)
                    p2.give_rew(1)

            else:
                # Player 2
                ok = False
                while not ok:
                    from_pos, slide = p2.choose_action(game)
                    ok = game.make_move(from_pos, slide)
                p2.add_state(game.convert_matrix_board_to_tuple(game.get_board()))

                win = game.check_winner()
                if win != -1:
                    if win == 0:
                        p1.give_rew(1)
                        p2.give_rew(0)
                    else:
                        p1.give_rew(0)
                        p2.give_rew(1)
    return p1, p2


def test(rl_player: RLPlayer, random_player, num_games):
    g = Game()
    player2_wins = 0
    games = 0
    for _ in range(num_games):
        winner = g.play(rl_player, random_player)
        games += 1
        g.reset()
        if winner == 1:
            player2_wins += 1

    print(f"RLPlayer won {player2_wins / num_games * 100}%")
