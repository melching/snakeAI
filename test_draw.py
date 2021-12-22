from snake import Snake
import time
import pygame

game = Snake(40, 40, render=True)
print(len(game.empty))

while game.state == "running":
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.move(0)
                print("Move Up")
            if event.key == pygame.K_RIGHT:
                game.move(1)
                print("Move Right")
            if event.key == pygame.K_DOWN:
                game.move(2)
                print("Move Down")
            if event.key == pygame.K_LEFT:
                game.move(3)
                print("Move Left")
            print("Empty Fields", len(game.empty))
            print("Snake", game.snake_body)

print(game.state)