# import numpy as np
import random
import pygame
import numpy as np

class Snake:
    def __init__(self,x_dim, y_dim, render=True):
        self.render = render

        self.state = "running"
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.snake_body = []
        self.apple = None
        self.empty = []
        for x in range(x_dim):
            for y in range(y_dim):
                self.empty.append((x,y))

        self.init_state(x_dim, y_dim)

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((800,800))
            self.draw_board()

    def init_state(self, x_dim, y_dim):
        assert x_dim>2 and y_dim>2, "Board is to small!"

        # set snake head
        head = (x_dim//2, y_dim//2)
        self.snake_body.append(head)
        self.empty.remove(head)

        # place apple
        self.place_apple_randomly()
        
    def place_apple_randomly(self,):
        rand_pos = random.choice(self.empty)
        self.apple = rand_pos
        self.empty.remove(self.apple)

    def place_apple(self, x, y):
        # removes old apple (manually for testing) and places new one
        assert (x,y) not in self.snake_body
        self.empty.append(self.apple)
        self.apple = (x,y)
        if self.render:
            self.draw_board()

    def move(self, direction):
        '''
        0 = up,
        1 = right,
        2 = down,
        3 = left
        '''
        assert self.state is not "failed", "Can't move, Game already ended!"

        diff_x = 0
        diff_y = 0
        if   direction == 0:
            diff_y = -1
        elif direction == 1:
            diff_x = 1
        elif direction == 2:
            diff_y = 1
        elif direction == 3:
            diff_x = -1

        nextpos = (self.snake_body[0][0] + diff_x, self.snake_body[0][1] + diff_y)

        # check if next pos hits border or body
        if nextpos[0] in [-1, self.x_dim] or nextpos[1] in [-1, self.y_dim] or nextpos in self.snake_body:
            self.state = "failed"
            return

        # check if next pos is apple
        if nextpos == self.apple:
            # insert nextpos to snake
            self.snake_body.insert(0, nextpos)
            # place new apple
            self.place_apple_randomly()

        # check if next pos is empty
        elif nextpos in self.empty:
            # move snake
            self.snake_body.insert(0,nextpos)
            # remove last pos
            self.empty.append(self.snake_body[-1])
            self.snake_body = self.snake_body[:-1]

        if self.render:
            self.draw_board()

    def get_board_as_numpy(self, dtype=np.uint8):
        '''
        0  = empty
        1  = snake body
        2  = snake head
        3  = apple
        '''
        board = np.zeros(shape=(self.x_dim, self.y_dim), dtype=dtype)
        board[self.snake_body[1:]] = 1
        board[self.snake_body[0]]  = 2
        board[self.apple]          = 3

        return board

    def get_score(self,):
        return len(self.snake_body)

    def quit_gui(self,):
        if self.render:
            pygame.quit()

    def draw_board(self,):
        assert self.render == True, "Draw has been calles while render is False!"

        board_color = (0,0,0)
        body_color  = (255,255,255)
        head_color  = (255,200,200)
        apple_color = (255,0,0)
        block_width = 15

        # draw board
        self.screen.fill((255,255,255))
        pygame.draw.rect(
            self.screen, 
            board_color, 
            (0 , 0, self.x_dim*block_width, self.y_dim*block_width), 
            0
        )

        # draw snake body and head
        for x,y in self.snake_body[1:]:
            pygame.draw.rect(
                self.screen, 
                body_color, 
                (x*block_width , y*block_width, block_width, block_width), 
                0
            )
        pygame.draw.rect(
            self.screen, 
            head_color, 
            (self.snake_body[0][0]*block_width , self.snake_body[0][1]*block_width, block_width, block_width), 
            0
        )

        # draw apple
        pygame.draw.rect(
            self.screen, 
            apple_color, 
            (self.apple[0]*block_width , self.apple[1]*block_width, block_width, block_width), 
            0
        )

        # show score
        pygame.display.set_caption('snakeAI - Current Score: ' + str(self.get_score()))

        # update window
        pygame.display.update()