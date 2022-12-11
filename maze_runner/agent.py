import collections
import operator
import numpy as np
import torch
import random
from collections import deque
from maze_game import BLOCK_SIZE, MazeRunnerGameAI, Direction, Point, index_of, PLAYERS_NUM,ARTBOARD_HEIGHT

from model import Linear_QNet, QTrainer
from helper import plot
from collections import defaultdict

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

NNRecord = collections.namedtuple("NNRecord", ["point", "distance"])
NNRecord.__doc__ = """
Used to keep track of the current best guess during a nearest
neighbor search.
"""

BT = collections.namedtuple("BT", ["value", "left", "right"])
BT.__doc__ = """
A Binary Tree (BT) with a node value, and left- and
right-subtrees.
"""

class GameFood:
    def __init__(self, xl, xr, yl, yr):
        self.xl = xl
        self.xr = xr
        self.yl = yl
        self.yr = yr

def SED(X, Y):
    return sum((i-j)**2 for i, j in zip(X, Y))

def kdtree(points):
    """Construct a k-d tree from an iterable of points.

    This algorithm is taken from Wikipedia. For more details,

    > https://en.wikipedia.org/wiki/K-d_tree#Construction

    """
    k = len(points[0])

    def build(*, points, depth):
        """Build a k-d tree from a set of points at a given
        depth.
        """
        if len(points) == 0:
            return None

        points.sort(key=operator.itemgetter(depth % k))
        middle = len(points) // 2

        return BT(
                value = points[middle],
                left = build(
                        points=points[:middle],
                        depth=depth+1,
                    ),
                right = build(
                        points=points[middle+1:],
                        depth=depth+1,
                    ),
        )

    return build(points=list(points), depth=0)

def find_nearest_neighbor(*, tree, point):
            """Find the nearest neighbor in a k-d tree for a given
            point.
            """
            k = len(point)

            best = None
            def search(*, tree, depth):
                """Recursively search through the k-d tree to find the
                nearest neighbor.
                """
                nonlocal best

                if tree is None:
                    return

                distance = SED(tree.value, point)
                if best is None or distance < best.distance:
                    best = NNRecord(point=tree.value, distance=distance)

                axis = depth % k
                diff = point[axis] - tree.value[axis]
                if diff <= 0:
                    close, away = tree.left, tree.right
                else:
                    close, away = tree.right, tree.left

                search(tree=close, depth=depth+1)
                if diff**2 < best.distance:
                    search(tree=away, depth=depth+1)

            search(tree=tree, depth=0)
            return best.point


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.8  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        self.min_dist =  [0 for x in range(PLAYERS_NUM)]


    def check_rect_collision(self, p, x, y, w, h):
        if p.x >= x and p.x <= x+w and p.y >= y and p.y <= y+h:
            # collision between p and rectangle
            return True
        return False

    def get_state(self, game, selected_player = 0):
        head = game.players[selected_player][0]
        aspect_ratio = game.maze.nx / game.maze.ny
        # Height and width of the maze image (excluding padding), in pixels
        height = ARTBOARD_HEIGHT
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / game.maze.ny, width / game.maze.nx
     
         


        reference_points_food = []


        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        target = 0
        # Game Food Target
        l = game.food[target].x < head.x  # food left
        r = game.food[target].x > head.x  # food right
        u = game.food[target].y < head.y  # food up
        d = game.food[target].y > head.y  # food down
     
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)[0]) or 
            (dir_l and game.is_collision(point_l)[0]) or 
            (dir_u and game.is_collision(point_u)[0]) or 
            (dir_d and game.is_collision(point_d)[0]),

            # Danger right
            (dir_u and game.is_collision(point_r)[0]) or 
            (dir_d and game.is_collision(point_l)[0]) or 
            (dir_l and game.is_collision(point_u)[0]) or 
            (dir_r and game.is_collision(point_d)[0]),

            # Danger left
            (dir_d and game.is_collision(point_r)[0]) or 
            (dir_u and game.is_collision(point_l)[0]) or 
            (dir_r and game.is_collision(point_u)[0]) or 
            (dir_l and game.is_collision(point_d)[0]),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Possible Movement
            l,
            r,
            u,
            d
            ]


        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = MazeRunnerGameAI()
    while True:
        # get old state
        def play(selected_player):
            state_old = agent.get_state(game, selected_player)
            # get move
            final_move = agent.get_action(state_old)
            # perform move and get new state
            reward, done, score = game.play_step(final_move, selected_player)
            state_new = agent.get_state(game, selected_player)
            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
            return done, score 
        
        all_scores = 0
        a_done = 0
        for i in range(PLAYERS_NUM):
            done, score = play(i)
            all_scores = score
            if done: 
                a_done = done
                break
      

        if a_done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if all_scores > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(all_scores)
            total_score += all_scores
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()