from snake import Snake
import time
import pygame

game = Snake(40, 40, render=True)

while game.state == "running":
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.move(0)
            if event.key == pygame.K_RIGHT:
                game.move(1)
            if event.key == pygame.K_DOWN:
                game.move(2)
            if event.key == pygame.K_LEFT:
                game.move(3)