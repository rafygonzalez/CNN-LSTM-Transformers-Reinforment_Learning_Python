
import pygame
from maze_game import MazeRunnerGameAI

if __name__ == '__main__':
    game = MazeRunnerGameAI()
    
    # game loop
    while True:
        reward, game_over, score = game.play_step(0)
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()