import random
import copy
import numpy as np
import time  # time

SIMULATION_TIMES = []  # time

WINNERS = {}


# No longer needed
def valid_state(state):
    valid = True
    for j in range(5):
        found_non_zero = False
        for i in range(6):
            if found_non_zero and state[i][j] == 0:
                valid = False
            if state[i][j] > 0:
                found_non_zero = True
    return valid


# No longer needed
def generate_random_state():
    state = []
    got_zero = [False for _ in range(5)]
    for _ in range(6):
        row = [random.randint(0, 2) if not got_zero[i]
               else 0 for i in range(5)]
        got_zero = [(got_zero[i] or row[i] == 0) for i in range(5)]
        state.append(row)
    state = state[::-1]
    return state


def get_actions(state):
    return [i for i in range(5) if state[0][i] == 0]


def apply_action(state, action, player):
    if state[0][action] > 0:
        print("Not a valid action")
        raise ValueError
    pos = -1
    for i in range(6):
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
    for i in range(6):
        for j in range(5-4+1):
            if np.unique(state_arr[i, j:j+4]).shape[0] == 1 and state_arr[i, j] > 0:
                WINNERS[hash] = state_arr[i, j]
                return state_arr[i, j]
    # vertical
    for i in range(6-4+1):
        for j in range(5):
            if np.unique(state_arr[i:i+4, j]).shape[0] == 1 and state_arr[i, j] > 0:
                WINNERS[hash] = state_arr[i, j]
                return state_arr[i, j]
    # diagonal top-left to bottom-right
    for i in range(6-4+1):
        for j in range(min(5-4+1, 6-4+1-i)):
            arr = [state_arr[i+k, j+k] for k in range(4)]
            if np.unique(arr).shape[0] == 1 and arr[0] > 0:
                WINNERS[hash] = arr[0]
                return arr[0]
    # diagonal bottom-left to top-right
    for i in range(3, 6):
        for j in range(min(5-4+1, i-6+4)):
            arr = [state_arr[i-k, j+k] for k in range(4)]
            if np.unique(arr).shape[0] == 1 and arr[0] > 0:
                WINNERS[hash] = arr[0]
                return arr[0]
    WINNERS[hash] = 0
    return 0


def get_hash(state):
    hash = 0
    for i in range(6):
        for j in range(5):
            hash += state[i][j]*3**(5*i+j)
    return hash


def get_state(hash):
    state = [[0 for __ in range(5)] for _ in range(6)]
    for i in range(6):
        for j in range(5):
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
        self.draw_reward= draw_reward

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
        start_time = time.time_ns()  # timeus
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


def game(player1, player2, verbose=False):
    state = [[0 for __ in range(5)] for _ in range(6)]
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


if __name__ == "__main__":    

    # np.random.seed(0)
    count_pl1=0
    count_pl2=0
    count_draw=0
    mc40 = Player(player=1, simulations=40)
    mc200 = Player(player=2, simulations=200)    
    #mc0 = Player(player=2, simulations=0)
    for _ in range(10):
        mc40 = Player(player=1, simulations=40)
        mc200 = Player(player=2, simulations=200)
        c = game(mc40, mc200)

        if(c==0):
            count_draw= count_draw + 1
        if(c==1):
            count_pl1 =  count_pl1 + 1
        if(c==2):
            count_pl2 =  count_pl2 + 1
    mc40.update_player(2)
    mc200.update_player(1)
    for _ in range(10):
        mc40 = Player(player=2, simulations=40)
        mc200 = Player(player=1, simulations=200)
        c= game(mc200, mc40)

        if(c==0):
            count_draw= count_draw + 1
        if(c==1):
            count_pl2 =  count_pl2 + 1
        if(c==2):
            count_pl1 =  count_pl1 + 1
    
    mc40.update_player(1)
    #game(mc40, mc200, verbose=True)
    print(  # time
        f"The average simulation time is {sum(SIMULATION_TIMES)/len(SIMULATION_TIMES)} nanoseconds")  # time
    print(f"Player 1 won: {count_pl1} times")
    print(f"Player 2 won: {count_pl2} times")
    print(f"Draws occurred {count_draw} times")
