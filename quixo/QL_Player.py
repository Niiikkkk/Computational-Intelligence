import pickle
import random
from game import Game, Move, Player
from tqdm.auto import tqdm


class QLPlayer(Player):
    def __init__(self, player, alpha, epsilon, discount_factor) -> None:
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.player = player
        self.states = []
        self.Q = {}

    def add_state(self, state, from_pos, slide):
        self.states.append((state, (from_pos, slide)))

    def get_Q_value(self, state, from_pos, slide):
        if (state, (from_pos, slide)) not in self.Q:
            self.Q[(state, (from_pos, slide))] = 0.0
        return self.Q[(state, (from_pos, slide))]

    def reset(self):
        self.states = []

    def choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        available_moves = game.get_possible_moves(self.player)
        Q_values = [self.get_Q_value(game.convert_matrix_board_to_tuple(game.get_board()), from_pos, slide) for
                    from_pos, slide in available_moves]

        if random.uniform(0, 1) < self.epsilon:  # do exploration for 30% of the time
            return random.choice(available_moves)
        else:  # take the best one for 70% of the time
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def give_rew(self, reward):
        for st in reversed(self.states):
            # st[0] = board state st[1][0] = from_pos st[1][1] = slice
            current_q_value = self.Q[(st[0], (st[1][0], st[1][1]))]
            reward = current_q_value + self.alpha * (self.discount_factor * reward - current_q_value)
            self.Q[(st[0], (st[1][0], st[1][1]))] = reward

    def save_policy(self, name):
        fw = open(name, 'wb')
        pickle.dump(self.Q, fw)
        fw.close()

    def load_policy(self, file):
        fr = open(file, 'rb')
        self.Q = pickle.load(fr)
        fr.close()


def train(p1: QLPlayer, p2: QLPlayer, game, epochs):
    for epoch in tqdm(range(epochs)):
        game.reset()
        p1.reset()
        p2.reset()
        state_board = game.convert_matrix_board_to_tuple(game.get_board())
        while game.check_winner() == -1:
            # Player 1
            ok = False
            while not ok:
                from_pos, slide = p1.choose_action(game)
                ok = game.make_move(from_pos, slide)
            p1.add_state(state_board, from_pos, slide)
            state_board = game.convert_matrix_board_to_tuple(game.get_board())

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
                p2.add_state(state_board, from_pos, slide)
                state_board = game.convert_matrix_board_to_tuple(game.get_board())

                win = game.check_winner()
                if win != -1:
                    if win == 0:
                        p1.give_rew(1)
                        p2.give_rew(0)
                    else:
                        p1.give_rew(0)
                        p2.give_rew(1)
    return p1, p2


def test(rl_player: QLPlayer, random_player, num_games):
    g = Game()
    player2_wins = 0
    games = 0
    for _ in range(num_games):
        winner = g.play(rl_player, random_player)
        games += 1
        print(games)
        g.reset()
        if winner == 1:
            player2_wins += 1

    print(f"RLPlayer won {player2_wins / num_games * 100}%")
