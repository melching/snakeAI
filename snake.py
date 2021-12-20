import numpy as np
import pygame

class Snake:
    '''
    -1 = border
    0  = empty
    1  = snake body
    2  = snake head
    3  = apple
    '''

    def __init__(self,x_dim, y_dim, render=True):
        self.render = render

        self.state = "running"
        self.score = 0

        self.create_board(x_dim, y_dim)

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((800,800))
            self.draw_board()

    def create_board(self, x_dim, y_dim):
        assert x_dim>2 and y_dim>2, "Board is to small!"

        # create emtpy board
        self.board = np.zeros(shape=(x_dim+2, y_dim+2), dtype=np.int8)
        
        # set boarders to -1
        self.board[0,:] = -1
        self.board[-1,:] = -1
        self.board[:,0] = -1
        self.board[:,-1] = -1

        # set snake head
        self.board[x_dim//2 + 1, y_dim//2 + 1] = 2

        # place apple
        self.place_apple()
        
    def place_apple(self,):
        # get empty fields
        where_empty = np.where(self.board==0)
        rand_pos = np.random.randint(where_empty[0].shape[0])
        self.board[where_empty[0][rand_pos], where_empty[1][rand_pos]] = 3

    def move(self,direction):
        '''
        0 = 
        '''

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
            (0 , 0, (self.board.shape[0]-2)*block_width, (self.board.shape[1]-2)*block_width), 
            0
        )

        # draw snake body and head
        where_body = np.where(self.board==1)
        for i in range(len(where_body[0])):
            pygame.draw.rect(
                self.screen, 
                body_color, 
                (where_body[0][i]*block_width , where_body[1][i]*block_width, block_width, block_width), 
                0
            )
        where_head = np.where(self.board==2)
        print(where_head)
        pygame.draw.rect(
            self.screen, 
            head_color, 
            (where_head[0][0]*block_width , where_head[1][0]*block_width, block_width, block_width), 
            0
        )

        # draw apple
        where_apple = np.where(self.board==3)
        pygame.draw.rect(
            self.screen, 
            apple_color, 
            (where_apple[0][0]*block_width , where_apple[1][0]*block_width, block_width, block_width), 
            0
        )

        # show score
        pygame.display.set_caption('snakeAI - Current Score: ' + str(self.score))

        # update window
        pygame.display.update()

