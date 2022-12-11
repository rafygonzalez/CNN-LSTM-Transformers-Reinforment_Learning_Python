from find_shortest_path import find_shortest_path
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from timeit import default_timer as timer
from maze import Maze
pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)


def index_of(val, in_list):
    try:
        return in_list.index(val)
    except ValueError:
        return -1


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20


ARTBOARD_WIDTH = 640
ARTBOARD_HEIGHT = 640

FOOD_NUM = 10
PLAYERS_NUM = 1
SPEED = 100
MAX_TIME = 120
MAX_COLLIDES = MAX_TIME * BLOCK_SIZE

class MazeRunnerGameAI:

    def __init__(self, w=ARTBOARD_WIDTH, h=ARTBOARD_HEIGHT):
        self.available_movement = []
        self.players_color = []


        self.start = timer()
        self.end = 0
        self.w = w
        self.h = h
        self.collision_points = []
        self.reward = 0
        self.direction = Direction.RIGHT
        self.n_collides = 0
      
        # Maze Generation and Data 
        # Maze dimensions (ncols, nrows)
        self.nx = 20
        self.ny = 20
        # Maze entry position
        self.ix = 0
        self.iy = 0

        # Food
        self.food = None
        self.food_num = FOOD_NUM
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Maze Runner')
        self.clock = pygame.time.Clock()

        self.score = 0
        self.level = 1
        self.frame_iteration = 0
        
        self.players = []
        self.players_head = []
        
        for _ in range(PLAYERS_NUM):
            self.players_color.append(self.random_color())
            self.available_movement.append([False, False, False, False])
            self.players_head.append(Point(8,8))

        for head in self.players_head:
            self.players.append([head])
            
 

        self.reset()

    def reset_maze_player_and_food(self):
        self.players = []
        self.collision_points = []
        self.players_head = []
        self.n_collides = 0
        self.maze = Maze(self.nx, self.ny, self.ix, self.iy)
        self.maze.make_maze()

        for _ in range(PLAYERS_NUM):
            self.players_head.append(Point(8,8))

        for head in self.players_head:
            self.players.append([head])

        self.food = []
        self.eaten = []
        self.place_food()

    def reset(self):
        # init game state
        self.start = timer()
        self.reward = 0
        self.direction = Direction.RIGHT
        self.score = 0
        self.frame_iteration = 0
        self.reset_maze_player_and_food()
        

    def check_rect_collision(self, p, x, y, w, h):
        if p.x >= x and p.x <= x+w and p.y >= y and p.y <= y+h:
            # collision between p and rectangle
            return True
        return False

    def place_food(self):
        aspect_ratio = self.nx / self.ny
        # Height and width of the maze image (excluding padding), in pixels
        height = ARTBOARD_HEIGHT
        width = int(height * aspect_ratio)
        scy, scx = height / self.ny, width / self.nx

        path = find_shortest_path(self.maze.maze_map, (0, 0), (19, 19))
        
        if path!= None:
            self.food_num = len(path)
            for p in path:
                foodPoint = Point(p[0] * scx, p[1] * scy)
                self.food.append(foodPoint)
            
    def calcWall(self, pt = None, selected_player=0):
        if pt == None:
            pt = self.players_head[selected_player]
        for point in self.collision_points:
                x1 = point[0] - BLOCK_SIZE
                x2 = point[2] - BLOCK_SIZE
                y1 = point[1] - BLOCK_SIZE
                y2 = point[3] - BLOCK_SIZE
                if x1 == x2:
                    for i in range(int(y1),int(y2)):
                        collision = self.check_rect_collision(pt,x2,i,BLOCK_SIZE,BLOCK_SIZE)
                        if collision:
                            self.available_movement[selected_player] = [
                                x2 < self.players_head[selected_player].x,
                                x2 > self.players_head[selected_player].x,
                                i < self.players_head[selected_player].y,  
                                i > self.players_head[selected_player].y           
                            ]
                            return [True, point]
               
                elif y1 == y2:
                    for i in range(int(x1),int(x2)):
                        collide = self.check_rect_collision(pt,i,y1,BLOCK_SIZE,BLOCK_SIZE)
                        if collide:
                            self.available_movement[selected_player] = [
                                i < self.players_head[selected_player].x,
                                i > self.players_head[selected_player].x,
                                y1 < self.players_head[selected_player].y,  
                                y1 > self.players_head[selected_player].y           
                            ]
                            return [True,point]
                else:
                    cornerxy0 = self.check_rect_collision(pt,x1,y1,BLOCK_SIZE,BLOCK_SIZE)
                    cornerxy1 = self.check_rect_collision(pt,x2,y2,BLOCK_SIZE,BLOCK_SIZE)
                    if cornerxy0 or cornerxy1:
                        self.available_movement[selected_player] = [
                                x2 < self.players_head[selected_player].x,
                                x2 > self.players_head[selected_player].x,
                                y2 < self.players_head[selected_player].y,  
                                y2 > self.players_head[selected_player].y           
                        ]
                        return [True, point]

        return [False, None]

    def is_collision(self, pt=None, selected_player=0):
       
        if pt == None:
            pt = self.players_head[selected_player]

        wallhit = self.calcWall(pt,selected_player)

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return [True, None]
        # hits itself
        for player in self.players:
            if pt in player[1:]:
                return [True, None]
        
        if wallhit[0]:
            return wallhit

        

        return [False, None]

    def draw_maze(self):
        aspect_ratio = self.nx / self.ny
        # Height and width of the maze image (excluding padding), in pixels
        height = ARTBOARD_HEIGHT
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx
        




        def write_wall(ww_x1, ww_y1, ww_x2, ww_y2):
            pygame.draw.line(self.display, BLUE1, (ww_x1, ww_y1), (ww_x2, ww_y2))

        def track(ww_x1, ww_y1, ww_x2, ww_y2):
            self.collision_points.append([ww_x1, ww_y1, ww_x2, ww_y2])
            
        self.collision_points = []
        for x in range(self.nx):
            for y in range(self.ny):
                if self.maze.cell_at(x, y).walls['S']:
                    x1, y1, x2, y2 = x * \
                        scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(x1, y1, x2, y2)
                    track(x1, y1, x2, y2)
                if self.maze.cell_at(x, y).walls['E']:
                    x1, y1, x2, y2 = (x + 1) * scx, y * \
                        scy, (x + 1) * scx, (y + 1) * scy
                    write_wall(x1, y1, x2, y2)
                    track(x1, y1, x2, y2)
    def random_color(self):
        rgbl=[255,0,0]
        random.shuffle(rgbl)
        return tuple(rgbl)
    def _update_ui(self):
        self.display.fill(BLACK)
        self.draw_maze()



        for pt in self.players:
            color = self.players_color[0]
            i = index_of(pt, self.players)
            if i != -1:
                color = self.players_color[i]

            aspect_ratio = self.nx / self.ny
            # Height and width of the maze image (excluding padding), in pixels
            height = ARTBOARD_HEIGHT
            width = int(height * aspect_ratio)
            # Scaling factors mapping maze coordinates to image coordinates
            scy, scx = height / self.ny, width / self.nx

       
            #if path != None:
            #    for p in path:
            #        pygame.draw.rect(self.display, RED, pygame.Rect(
            #                    p[0] * scx, p[1] * scx, 15, 15))

            pygame.draw.rect(self.display, color, pygame.Rect(
                pt[0].x, pt[0].y, BLOCK_SIZE, BLOCK_SIZE))
            

        for food in self.food:
            exists = index_of(food, self.eaten)
            if exists == -1:
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))


        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        record = font.render(
            "Time Elapse: " + str(int(timer() - self.start)), True, WHITE)
        self.display.blit(record, [0, 30])
        level = font.render(
            "level: " + str(self.level), True, WHITE)
        self.display.blit(level, [ARTBOARD_WIDTH - 100, 0])
        pygame.display.flip()

    def play_step(self, action = None, selected_player = 0):

        self.frame_iteration += 1
        # 1. collect user input
        game_over = False

       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        currentTime = timer()
        timeExceeded = currentTime - self.start > MAX_TIME
        iterationException = self.frame_iteration > 100 * ARTBOARD_HEIGHT

        if timeExceeded or iterationException:
            game_over = True
            self.reward = -15
            return self.reward, game_over, self.score

        if action != None:
            cachedPlayer = self.players[selected_player]
            cachedHead = self.players_head[selected_player]
            self._move(action, selected_player) # update the head
            collisionData = self.is_collision(self.players_head[selected_player], selected_player)
            collide = collisionData[0]
            if collide:
                self.players[selected_player] = cachedPlayer
                self.players_head[selected_player] = cachedHead
                self.n_collides += 1
                self.reward = -15
                if self.n_collides >= MAX_COLLIDES:
                    game_over = True
                    self.reward = -50
            else: 
                self.players[selected_player].insert(0, self.players_head[selected_player])
                self.players[selected_player].pop()
             
            
            for food in self.food:
                collides = self.check_rect_collision(self.players[selected_player][0],food.x,food.y,BLOCK_SIZE,BLOCK_SIZE)
                if collides or self.players_head[selected_player] == food:
                   exists = index_of(food, self.eaten)
                   if exists == -1:
                      self.score += 1
                      self.reward = 15
                      self.eaten.append(food)
                      self.food.remove(food)
                      self.start = timer()
                      self.n_collides = 0



            if len(self.eaten) == self.food_num:
                self.level += 1
                self.reward = 30
                self.reset_maze_player_and_food()
    
            # 5. update ui and clock
            self._update_ui()
            self.clock.tick(SPEED)
            # 6. return game over and score
            return self.reward, game_over, self.score
        else:
            cachedPlayer = self.players[selected_player]
            cachedHead = self.players_head[selected_player]
            self._move(None, selected_player)
            collisionData = self.is_collision(selected_player)
            collide = collisionData[0]
            if collide:
                self.players[selected_player] = cachedPlayer
                self.players_head[selected_player] = cachedHead
                self.reward = -15
            else: 
                self.players[selected_player].insert(0, self.players_head[selected_player])
                self.players[selected_player].pop()
                self.reward = 0


            # 5. update ui and clock
            if len(self.eaten) == FOOD_NUM:
                self.end = timer()
                self.place_food()

            self._update_ui()
            self.clock.tick(SPEED)
            # 6. return game over and score
            return self.reward, game_over, self.score

    def _move(self, action = None, selected_player = 0):
        if action:
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx] # no change
            elif np.array_equal(action, [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
            else: # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

            self.direction = new_dir

            x = self.players_head[selected_player].x
            y = self.players_head[selected_player].y
            moveN = BLOCK_SIZE / 2
            if self.direction == Direction.RIGHT:
                    x += moveN
            elif self.direction == Direction.LEFT:
                    x -= moveN
            elif self.direction == Direction.DOWN:
                    y += moveN
            elif self.direction == Direction.UP:
                    y -= moveN

            self.players_head[selected_player] = Point(x, y)
        else:
            # [straight, right, left]
            x = self.players_head[selected_player].x
            y = self.players_head[selected_player].y
            moveN = BLOCK_SIZE / 2
            if self.direction == Direction.RIGHT:
                x += moveN 
            elif self.direction == Direction.LEFT:
                x -= moveN 
            elif self.direction == Direction.DOWN:
                y += moveN 
            elif self.direction == Direction.UP:
                y -= moveN 
            self.players_head[selected_player] = Point(x, y)
