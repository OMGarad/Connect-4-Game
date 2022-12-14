
##############

#Your program can go here.

###############
import random
import copy
import numpy as np
import time  # time
import matplotlib.pyplot as plt
import gzip

SIMULATION_TIMES = [0]  # time

WINNERS = {}

r, c = 4, 5


def valid_state(state):  # No longer needed
    valid = True
    for j in range(c):
        found_non_zero = False
        for i in range(r):
            if found_non_zero and state[i][j] == 0:
                valid = False
            if state[i][j] > 0:
                found_non_zero = True
    return valid


def generate_random_state():  # No longer needed
    state = []
    got_zero = [False for _ in range(c)]
    for _ in range(r):
        row = [random.randint(0, 2) if not got_zero[i]
               else 0 for i in range(c)]
        got_zero = [(got_zero[i] or row[i] == 0) for i in range(c)]
        state.append(row)
    state = state[::-1]
    return state


def get_actions(state):
    return [i for i in range(c) if state[0][i] == 0]


def apply_action(state, action, player):
    if state[0][action] > 0:
        print("Not a valid action")
        raise ValueError
    pos = -1
    for i in range(r):
        if state[i][action] == 0:
            pos = i
        else:
            break
    new_state = copy.deepcopy(state)
    new_state[pos][action] = player
    return new_state


def get_winner(hash):
    if hash in WINNERS:
        return WINNERS[hash]
    state_arr = np.array(get_state(hash))
    # horizontal
    for i in range(r):
        for j in range(c-4+1):
            if np.unique(state_arr[i, j:j+4]).shape[0] == 1 and state_arr[i, j] > 0:
                WINNERS[hash] = state_arr[i, j]
                return state_arr[i, j]
    # vertical
    for i in range(r-4+1):
        for j in range(c):
            if np.unique(state_arr[i:i+4, j]).shape[0] == 1 and state_arr[i, j] > 0:
                WINNERS[hash] = state_arr[i, j]
                return state_arr[i, j]
    # diagonal top-left to bottom-right
    for i in range(r-4+1):
        for j in range(min(c-4+1, r-4+1-i)):
            arr = [state_arr[i+k, j+k] for k in range(4)]
            if np.unique(arr).shape[0] == 1 and arr[0] > 0:
                WINNERS[hash] = arr[0]
                return arr[0]
    # diagonal bottom-left to top-right
    for i in range(3, r):
        for j in range(min(c-4+1, i-2)):
            arr = [state_arr[i-k, j+k] for k in range(4)]
            if np.unique(arr).shape[0] == 1 and arr[0] > 0:
                WINNERS[hash] = arr[0]
                return arr[0]
    WINNERS[hash] = 0
    return 0


def get_hash(state):
    hash = 0
    for i in range(r):
        for j in range(c):
            hash += state[i][j]*3**(c*i+j)
    return hash


def get_state(hash):
    state = [[0 for __ in range(c)] for _ in range(r)]
    for i in range(r):
        for j in range(c):
            state[i][j] = hash % 3
            hash //= 3
    return state


def get_player(state):
    val = sum([sum(row) for row in state])
    if val % 3 == 0:
        return 1
    return 2


class Node:
    def __init__(self, hash):
        self.hash = hash
        self.state_set = False
        self.actions_set = False
        self.player_set = False
        self.winner_set = False
        self.children_set = False
        self.state = None
        self.actions = None
        self.player = None
        self.winner = None
        self.children = None
        self.reward = 0
        self.num = 0

    def fetch_state(self):
        if not self.state_set:
            self.state = get_state(self.hash)
            self.state_set = True
        return self.state

    def fetch_actions(self):
        if not self.actions_set:
            self.actions = get_actions(self.fetch_state())
            self.actions_set = True
        return self.actions

    def fetch_player(self):
        if not self.player_set:
            self.player = get_player(self.fetch_state())
            self.player_set = True
        return self.player

    def fetch_winner(self):
        if not self.winner_set:
            self.winner = get_winner(self.hash)
            self.winner_set = True
        return self.winner

    def fetch_children(self):
        if not self.children_set:
            if self.fetch_winner() == 0:
                state = self.fetch_state()
                player = self.fetch_player()
                self.children = [get_hash(apply_action(
                    state, action, player)) for action in self.fetch_actions()]
            else:
                self.children = []
            self.children_set = True
        return self.children


class Player:
    def __init__(self, player, simulations, win_reward=5, draw_reward=1):
        self.node_map = {}
        self.player = player
        self.simulations = simulations
        self.win_reward = win_reward
        self.draw_reward = draw_reward

    def update_player(self, player):
        self.player = player

    def selection(self, root):
        path = []
        node = root
        while True:
            path.append(node)
            if node.num == 0:  # If node unexplored
                break
            if node.fetch_winner() > 0:  # If winning state already found
                break
            unexplored_child = False
            for ch in node.fetch_children():  # If any child unexplored
                self.node_map[ch] = self.node_map.get(ch, Node(ch))
                if self.node_map[ch].num == 0:
                    unexplored_child = True
                    node = self.node_map[ch]
                    break
            if unexplored_child:
                continue
            if len(node.fetch_children()) == 0:  # If draw state
                break
            # Choose max value
            node = self.node_map[
                node.fetch_children()[
                    np.argmax(
                        [self.node_map[ch].reward/self.node_map[ch].num+ (100) *
                         (np.log(node.num)/self.node_map[ch].num)**0.5 for ch in node.fetch_children()]
                    )
                ]
            ]
        return node, path[::-1]

    def random_moves(self, node):
        winner = 0
        n = node
        while True:
            if n.fetch_winner() > 0:
                winner = n.fetch_winner()
                break
            if len(n.fetch_children()) == 0:  # if draw
                break
            ch = np.random.choice(n.fetch_children())
            self.node_map[ch] = self.node_map.get(ch, Node(ch))
            n = self.node_map[ch]
        return winner

    def back_propogation(self, path, winner):
        for n in path:
            n.num += 1
            if winner == self.player:
                n.reward += self.win_reward
            elif winner == 0:
                n.reward += self.draw_reward
            else:
                n.reward -= self.win_reward

    def simulation(self, root):
        start_time = time.time_ns()  # time
        node, path = self.selection(root)
        # print(node.hash)
        for ch in node.fetch_children():
            self.node_map[ch] = self.node_map.get(ch, Node(ch))
        winner = self.random_moves(node)
        self.back_propogation(path, winner)
        end_time = time.time_ns()  # time
        SIMULATION_TIMES.append(end_time-start_time)  # time

    def simulate(self, root):
        for _ in range(self.simulations):
            self.simulation(root)

    def get_best_play(self, state):
        hash = get_hash(state)
        self.node_map[hash] = self.node_map.get(hash, Node(hash))
        root = self.node_map[hash]
        for ch in root.fetch_children():
            self.node_map[ch] = self.node_map.get(ch, Node(ch))

        self.simulate(root)

        all_children_explored = True
        for ch in root.fetch_children():
            if self.node_map[ch].num == 0:
                all_children_explored = False
        if all_children_explored:
            action = root.fetch_actions()[
                np.argmax(
                    [self.node_map[ch].reward /
                        self.node_map[ch].num for ch in root.fetch_children()]
                )
            ]
        else:
            unexplored_indices = []
            for i, ch in enumerate(root.fetch_children()):
                if self.node_map[ch].num == 0:
                    unexplored_indices.append(i)
            action = root.fetch_actions()[np.random.choice(unexplored_indices)]

        return action

    def clear_tree(self):
        self.node_map = {}


class Q_Learning:
    def __init__(self, player, win_reward = 5, draw_reward =2, epsilon=0, alpha=0.05, gamma=0.9, init_row=np.array([ 1, 1, 1, 1, 1])): # np.zeros((c))
        self.table = {}
        self.player = player
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.init_row = init_row
        self.last_state = None
        self.r = {}
    
    def get_r(self,hash):
        if hash not in self.r:
            if get_winner(hash) == self.player:
                self.r[hash] = self.win_reward
            elif get_winner(hash) > 0:
                self.r[hash] = -self.win_reward
            elif len(get_actions(get_state(hash))) == 0:
                self.r[hash] = self.draw_reward
            else:
                self.r[hash] = 0
        return self.r[hash]
    
    def update_values(self, hash, action, hash_next):
        """ hash_next = get_hash(apply_action(
            get_state(hash), action, self.player)) """
        self.table[hash] = self.table.get(hash, self.init_row.copy())
        self.table[hash_next] = self.table.get(hash_next, self.init_row.copy())
        self.table[hash][action] += self.alpha * \
            (self.get_r(hash_next) + self.gamma *
             np.argmax(self.table[hash_next]) - self.table[hash][action])

    def get_best_play(self, state):
        hash = get_hash(state)

        if self.last_state is not None:
            self.update_values(self.last_state[0], self.last_state[1], hash)

        actions = get_actions(state)
        if np.random.random() < self.epsilon:
            action = np.random.choice(actions)
            #action = np.random.choice([act for act in actions if self.table[hash][act] == 1])
        else:
            self.table[hash] = self.table.get(hash, self.init_row.copy())
            action = actions[np.argmax(self.table[hash][actions])]

        self.last_state = (hash, action)

        newHash = get_hash(apply_action(get_state(hash), action, self.player))
        if get_winner(newHash) != 0:
            self.table[hash][action] = self.get_r(newHash)

        return action


def display(state):
    string = ""
    for row in state:
        for e in row:
            if e == 0:
                string += "."
            elif e == 1:
                string += "X"
            else:
                string += "O"
        string += "\n"
    string += "\n"
    print(string)


def write_dat_file(ql):
    with open("2019A7PS1010G_OMKAR.dat", "w") as file:
        for key in ql.table.keys():
            s = ""
            s += str(key) + " "
            s += " ".join([str(i) for i in ql.table[key].tolist()])
            s += "\n"
            file.write(s)

def read_dat_file():
    table = dict()
    with open("2019A7PS1010G_OMKAR.dat", "r") as file:
        for line in file.readlines():
            a, *b = line.split()
            key = int(a)
            array = np.array([float(i) for i in b])
            table[key] = array
    return table


def game_mcts(player1, player2, verbose=False):
    state = [[0 for __ in range(c)] for _ in range(6)]
    while get_winner(get_hash(state)) == 0 and len(get_actions(state)) > 0:
        if get_player(state) == 1:
            action = player1.get_best_play(state)
        else:
            action = player2.get_best_play(state)
        if verbose:
            print(f"Player {get_player(state)} plays action: {action}")
        state = apply_action(state, action, get_player(state))
        if verbose:
            display(state)
    if get_winner(get_hash(state))>0:
        print(f"The winner is player {get_winner(get_hash(state))}")
    else:
        print("The game was a draw")
    player1.clear_tree()
    player2.clear_tree()
    
    return get_winner(get_hash(state))

def game_ql(player1, player2, verbose=False):
    state = [[0 for __ in range(c)] for _ in range(4)]
    while get_winner(get_hash(state)) == 0 and len(get_actions(state)) > 0:
        if get_player(state) == 1:
            action = player1.get_best_play(state)
        else:
            action = player2.get_best_play(state)
        if verbose:
            print(f"Player {get_player(state)} plays action: {action}")
        state = apply_action(state, action, get_player(state))
        if verbose:
            display(state)
    if get_winner(get_hash(state)) == 1:
        player2.table[player2.last_state[0]][player2.last_state[1]] = player2.get_r(get_hash(state))
    if get_winner(get_hash(state))>0:
        print(f"The winner is player {get_winner(get_hash(state))}")
    else:
        print("The game was a draw")
    player1.clear_tree()

    return get_winner(get_hash(state))

def graph_variation(x,y):
    plt.xlabel('Value of Alpha')
    plt.ylabel('Percentage of wins')
    plt.title('Average Win Percentage of Q-Learning v/s Value of Alpha')

    plt.plot(x,y, marker='o')

    plt.show()
    return    


def graph(x, y):
    plt.xlabel('No. of runs')
    plt.ylabel('Percentage of wins')
    plt.title('MC0 v/s Q-Learning (Random Action Modification)')
    
    for yi in y:
        plt.plot(x, yi)
    plt.legend(['Draw', 'MC_0', 'Q-Learning'])
    
    plt.show()
    return

#if __name__ == "__main__":
    
def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

def main():
    
    choice = input("Choose a or c: ")

    mc200= Player(player=1, simulations=200)
    mc40=Player(player=2, simulations=40)
    mcn =Player(player=1, simulations=0)
    ql = Q_Learning(player=2)
    global r
    if choice=="a":
        r=6
        game_mcts(mc200, mc40, verbose=True)
    elif choice=="c":
        r=4
        ql.table= read_dat_file()
        game_ql(mcn, ql, verbose=True)
        print("Maximum value of n and r are: 0 and 4")
      
    """
    # np.random.seed(0)
    #input()
    begin= time.monotonic()
    
    ql = Q_Learning(player=2)
    #ql.table= read_dat_file()
    #total_count = [0,0,0]
    runs=1000
    games_per_run=100
    y = [[], [], []]
    #alpha_values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.6, 0.7, 0.8, 0.9]

  
    win_perc= []
    for al in alpha_values:
        print(f"The value of alpha is {al}")
        ql.epsilon = 0.1
        ql.alpha= al
        it=0
        count = [0,0,0]
        
        for j in range(runs*games_per_run + 1000):
            mcn = Player(player=1, simulations=0)
            game_ql(mcn, ql)
            if(j%500==0 and j!=0 and j<runs*games_per_run):
                ql.epsilon=ql.epsilon*0.75
                print(f"Run {j} and epsilon value is {ql.epsilon}")
            it+=1
            if it>=runs*games_per_run:
                count[game_ql(mcn, ql)] += 1
        win_perc.append((count[2]/1000)*100)
    
    print(win_perc)
    print(alpha_values)

    graph_variation(alpha_values, win_perc)
  

    print(count)
        for i, cnt in enumerate(count):
            y[i].append((cnt / games_per_run)*100)

    for i in range(runs):
        count = [0,0,0]
        if(i%5==0 and i!=0):
            ql.epsilon=ql.epsilon*0.5
        print(f"Run {i} and epsilon value is {ql.epsilon}")
        
        for j in range(games_per_run):
            mcn = Player(player=1, simulations=0)
            count[game_ql(mcn, ql)] += 1
        print(count)
        for i, cnt in enumerate(count):
            y[i].append((cnt / games_per_run)*100)


    #print(len(ql.table))

    write_dat_file(ql)

    print(time.monotonic()- begin)
    graph([x for x in range(0, runs * games_per_run, games_per_run)], y)
    
    #for i in range(3):
    #    total_count[i]=(total_count[i]/runs)*100
    #print(total_count)
    #print(total_count[2]- total_count[1])



    count = [0,0,0]
    mc40 = Player(player=1, simulations=40)
    mc200 = Player(player=2, simulations=200)
    # mc0 = Player(player=2, simulations=0)
    for _ in range(10):
        mc40 = Player(player=1, simulations=40)
        #if(_==0):
        #    mc40.simulate()
        mc200 = Player(player=2, simulations=200)
        count[game_mcts(mc40, mc200)] += 1
    mc40.update_player(2)
    mc200.update_player(1)
    for _ in range(10):
        mc40 = Player(player=2, simulations=40)
        mc200 = Player(player=1, simulations=200)
        #if(_==0):
        #   mc200.simulate()
        count[game_mcts(mc200, mc40)] += 1
    mc40.update_player(1)
    #game(mc40, ql, verbose=True)
    print(  # time
        f"The average simulation time is {sum(SIMULATION_TIMES)/len(SIMULATION_TIMES)} nanoseconds")  # time
    print(count)
    print(time.monotonic()- begin)

    """
    """ print("************ Sample output of your program *******")

    game1 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [0,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]


    game2 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [1,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]

    
    game3 = [ [0,0,0,0,0],
              [0,0,0,0,0],
              [0,2,1,0,0],
              [1,2,2,0,0],
              [1,1,2,2,0],
              [2,1,1,1,2],
            ]

    print('Player 2 (Q-learning)')
    print('Action selected : 2')
    print('Value of next state according to Q-learning : .7312')
    PrintGrid(game1)


    print('Player 1 (MCTS with 25 playouts')
    print('Action selected : 1')
    print('Total playouts for next state: 5')
    print('Value of next state according to MCTS : .1231')
    PrintGrid(game2)

    print('Player 2 (Q-learning)')
    print('Action selected : 2')
    print('Value of next state : 1')
    PrintGrid(game3)
    
    print('Player 2 has WON. Total moves = 14.')
    """ 
if __name__=='__main__':
    main()