import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from timeit import default_timer as timer

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
SPEED = 80

MAX_TIME = 2
FOOD_NUM = 5

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.start = timer()
        self.end = 0
        self.w = w
        self.h = h
        self.eaten = []
        self.food_num = FOOD_NUM
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):

        ##if self.food: 
        ##    return
    
        self.food = []
        self.eaten = []

        for _ in range(self.food_num):
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            food = Point(x, y)
            self.food.append(food)


        for food in self.food:
            if food in self.snake:
                self._place_food()


            
     
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False

        iterationException = self.frame_iteration > 100*len(self.snake)
        collide = self.is_collision()
        currentTime = timer()
        timeExceeded = currentTime - self.start > MAX_TIME


        if collide or iterationException or timeExceeded:
            reward = -15
            game_over = True
            self.start = timer()
            return reward, game_over, self.score


        # 4. place new food or just move

        shouldDelete = True
        for food in self.food:
              if self.head == food:
                 exists = index_of(food, self.eaten)
                 if exists == -1:
                    self.score += 1
                    reward = 15
                    shouldDelete = False
                    self.eaten.append(food)
                    self.food.remove(food)
                    self.start = timer()
                    

        if shouldDelete:
             self.snake.pop()

        if len(self.eaten) == FOOD_NUM:
            self.end = timer()
            self._place_food()
      

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        for food in self.food:
            exists = index_of(food, self.eaten)
            if exists == -1:
                pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))



    
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])

        record = font.render("Time Elapse: " + str(int(timer() - self.start)) , True, WHITE)
        self.display.blit(record, [0, 30])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

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

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)