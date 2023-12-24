import random
from game import Game, Move, Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move
    
    def give_rew(self, reward):
        pass

    def add_state(self,s):
        pass
    
class RLPlayer(Player):
    def __init__(self,player,esp_rate = 0.3,lr = 0.1) -> None:
        super().__init__()
        self.esp_rate = esp_rate
        self.lr = lr
        self.player=player
        self.states = []
        self.state_value = {}

    def add_state(self,s):
        self.states.append(s)
    
    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        if(random.random() <= self.esp_rate):  #do exploration for 30% of the time
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            move = (from_pos,move)
        else: #take the best one for 70% of the time
            pos = game.get_possible_moves(self.player)
            value_max = -999
            for p in pos:
                tmp = game.get_board()
                game.move(p[0],p[1],self.player)
                next_board = game.get_hash_board()
                game.set_board(tmp)
                value_act = 0 if self.state_value.get(next_board) is None else self.state_value.get(next_board) 
                if value_act > value_max:
                    value_max = value_act
                    move = p
        return move

    def give_rew(self, reward):
        for v in reversed(self.states):
            if self.state_value.get(v) is None:
                self.state_value[v] = 0
            self.state_value[v] += self.lr*(reward-self.state_value[v])
            reward = self.state_value[v]

def train(p1,p2,game,epochs):
    for epoch in range(epochs):
        if epoch % 1000 == 0:
            print("Epoch:",epoch)
        game.play(p1,p2)
        game.reset()

if __name__ == '__main__':
    g = Game()
    player1 = RLPlayer(player=0)
    player2 = RLPlayer(player=1)
    train(player1, player2,g,epochs=2000)

    g = Game()
    player1 = RandomPlayer()
    player2_wins = 0
    for _ in range(100):
        winner = g.test(player1, player2)
        g.reset()
        if winner == 1:
            player2_wins+=1

    print(f"RLPlayer won {player2_wins}%")