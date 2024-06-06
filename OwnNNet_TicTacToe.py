import pygame as pyg
import math
import sys
from utilities import constants as c
from entities import Memory, Game, agent
from recordtype import recordtype
from itertools import count
import matplotlib.pyplot as plt
import numpy as np

class PlayerSprite(pyg.sprite.Sprite):
    def __init__(self, row, col, turn=None, image=None):
        pyg.sprite.Sprite.__init__(self)
        if image:
            self.image = pyg.transform.scale(image, (round(c.WIDTH / (2*c.N)), round(c.HEIGHT / (2*c.N))))
        else:
            self.image = pyg.transform.scale(images[turn - 1], (round(c.WIDTH / (2*c.N)), round(c.HEIGHT / (2*c.N))))
        self.image.set_colorkey(c.BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (c.WIDTH / (c.N + 2) * (col + 1) + c.WIDTH / ((c.N+2)*2),
                            c.HEIGHT / (c.N + 2) * (row + 1) + c.HEIGHT / ((c.N+2)*2))

def draw_text(surf, text, size, x, y, color=(255, 255, 255)):
    font = pyg.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    surf.blit(text_surface, text_rect)


def draw_background():
    for i in range(2, c.N+1, 1):
        pyg.draw.line(screen, c.GREY,
                      (c.WIDTH / (c.N+2) * i, c.HEIGHT / (c.N+2)),
                      (c.WIDTH / (c.N+2) * i, c.HEIGHT / (c.N+2) * (c.N+1)), 11)
        pyg.draw.line(screen, c.GREY,
                      (c.WIDTH / (c.N+2), c.HEIGHT / (c.N+2) * i),
                      (c.WIDTH / (c.N+2) * (c.N+1), c.HEIGHT / (c.N+2) * i), 11)

def pygame_loop():
    # Game loop
    running = True
    while running:
        clock.tick(c.FPS)
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                running = False
            elif event.type == pyg.MOUSEBUTTONDOWN:
                x, y = pyg.mouse.get_pos()
                if (c.WIDTH / (c.N+2)) < x < (c.WIDTH / (c.N+2) * (c.N+1)) and (c.HEIGHT / (c.N+2)) < y < (c.HEIGHT / (c.N+2) * (c.N+1)):
                    col = math.floor(x / (c.WIDTH / (c.N+2)) - 1)
                    row = math.floor(y / (c.HEIGHT / (c.N+2)) - 1)

                    ## if the cell is occupied by the opponent, or user already
                    ## or it is the middle cell, do nothing.
                    if game.state[row,col] != 0: # or (row == (c.N-1)/2 and col == (c.N-1)/2):
                        continue
                       
                    termination_state, sprite_params = game.new_play(row, col)
                    game.PLAY_SPRITES.append(
                        PlayerSprite(sprite_params[0], sprite_params[1], sprite_params[2]))
                    all_sprites.add(game.PLAY_SPRITES[-1])
                    all_sprites.draw(screen)
                    pyg.display.flip()
                    if termination_state == -1:
                        game_over_screen(game.winner)
            elif event.type == pyg.KEYDOWN:
                key_state = pyg.key.get_pressed()
                if key_state[pyg.K_ESCAPE]:
                    running = False
        # Update
        ## block the middle cell using an icon if N is odd
        if c.N%2==1:
            block_image = pyg.image.load('media/blocked.png').convert()
            all_sprites.add(PlayerSprite((c.N-1)/2, (c.N-1)/2, image=block_image))
        all_sprites.update()

        # Draw / render
        screen.fill(c.BLACK)
        draw_background()
        previous_turn = game.turn

        ## set the size of the rectangles around the scores shown on corners equal to cell size
        if game.turn == agent.adversary:
            pyg.draw.rect(screen, c.RED,
                          (1, ## x of upper corner of the rect
                           1, ## y of upper corner of the rect
                           round(c.WIDTH / (c.N + 2)), ## width of the rect (now it is equal to cell size)
                           round(c.HEIGHT / (c.N + 2))), ## height of the rect
                          2,
                          5)
        else:
            pyg.draw.rect(screen, c.CYAN,
                          (round(c.WIDTH / (c.N + 2) * (c.N + 1)), ## the rect starts where the cells end
                           1,
                           round(c.WIDTH / (c.N + 2)), ## width of the rect (now it is equal to cell size)
                           round(c.HEIGHT / (c.N + 2))), ## height of the rect
                           2,
                           5)
            
            if c.HUMAN_VS_AI:
                termination_state, sprite_params = game.AI_play()
            else:
                termination_state, sprite_params = agent.play_visual(previous_turn, game)
            game.PLAY_SPRITES.append(PlayerSprite(sprite_params[0], sprite_params[1], sprite_params[2]))
            all_sprites.add(game.PLAY_SPRITES[-1])
            all_sprites.draw(screen)
            pyg.display.flip()
            if termination_state == -1:
                game_over_screen(game.winner)
        
        ## Displaying scores: adaptive size
        draw_text(screen, str(game.score[0]), int(c.WIDTH/(c.N+2)),1 + c.WIDTH/((c.N+2)*2) , 1 + c.WIDTH/((c.N+2)*2), c.RED)
        draw_text(screen, str(game.score[1]), int(c.WIDTH/(c.N+2)), c.WIDTH / (c.N + 2) * (c.N + 1) +1 + c.WIDTH/((c.N+2)*2), 1 + c.WIDTH/((c.N+2)*2), c.CYAN)
        all_sprites.draw(screen)
        pyg.display.flip()


def game_over_screen(winner):
    s = pyg.Surface((c.WIDTH, c.HEIGHT), pyg.SRCALPHA)
    s.fill((64, 64, 64, 164))
    screen.blit(s, (0, 0))
    draw_text(screen, "Game Over!", 64, c.WIDTH / 2, c.HEIGHT / 4, c.YELLOW)
    if winner == 3:
        draw_text(screen, "It was a tie! ", 32,
                  c.WIDTH / 2, c.HEIGHT / 2, c.YELLOW)
    else:
        draw_text(screen, "Player " + str(winner) + ' won', 32,
                  c.WIDTH / 2, c.HEIGHT / 2, c.RED if winner == 1 else c.CYAN)
        game.score[winner - 1] += 1
    draw_text(screen, "<Press any key to restart>", 24, c.WIDTH / 2, c.HEIGHT * 3 / 4, c.YELLOW)
    pyg.display.flip()
    waiting = True
    while waiting:
        clock.tick(c.FPS)
        for event in pyg.event.get():
            if event.type == pyg.QUIT:
                pyg.quit()
                sys.exit()
            if event.type == pyg.KEYDOWN or event.type == pyg.MOUSEBUTTONDOWN:
                key_state = pyg.key.get_pressed()
                if key_state[pyg.K_ESCAPE]:
                    pyg.quit()
                    sys.exit()
                waiting = False
                game.new_game()


def play_wo_training(game, agent):
    wins = 0
    looses = 0
    ties = 0
    i_episode = 0
    game.new_game()
    for i_episode in range(c.NUM_GAMES):
        for _ in count():
            previous_turn = game.turn
            if game.turn == agent.adversary:
                termination_state, _ = game.AI_play()
            else:
                termination_state, _ = agent.play_visual(previous_turn, game)

            if termination_state == -1:
                # print(game.state, game.winner)
                wins += 1 if game.winner == agent.NNet_player else 0
                looses += 1 if game.winner == agent.adversary else 0
                ties += 1 if game.winner == 3 else 0
                game.new_game()
                break
    print('After ', i_episode, ' games')
    print('w: ', wins, ' l:', looses, ' t:', ties)


def silent_training(game, agent, replay_memory):
    total_losses = []
    iteration = 0
    loss = [0]
    total_illegal_moves = []
    wins = 0
    total_wins = []
    looses = 0
    ties = 0
    game.new_game()
    for i_episode in range(c.NUM_GAMES):
        illegal_moves = 0
        for _ in count():
            iteration += 1
            previous_turn = game.turn
            if game.turn == agent.adversary:
                termination_state, _ = game.AI_play()
                if game.winner > 0:
                    replay_memory.memory[-1].reward =\
                        agent.calculate_reward(previous_turn, game.turn, game.winner)
                if len(replay_memory.memory) > 0:
                    replay_memory.memory[-1].next_state = game.state.copy()
            else:
                termination_state, _, illegal_moves =\
                    agent.play(previous_turn, game, replay_memory, experience, illegal_moves)

            # if Game over, update counters and start a new game
            if termination_state == -1:
                if game.winner == agent.NNet_player:
                    wins += 1
                    total_wins.append(1)
                else:
                    looses += 1 if game.winner == agent.adversary else 0
                    ties += 1 if game.winner == 3 else 0
                    total_wins.append(0)
                game.new_game()
                break

        # If we have enough experiences, start optimizing
        if replay_memory.can_sample_memory(c.BATCH_SIZE * c.EPOCHS):
            experiences = replay_memory.sample(c.BATCH_SIZE * c.EPOCHS)
            loss = agent.PolicyNetwork.RL_train(experiences, agent.TargetNetwork, experience, iteration)
            total_losses.append(loss)

        if i_episode % c.TARGET_UPDATE == 0:
            agent.TargetNetwork.copy_from(agent.PolicyNetwork)

        if i_episode % 27 == 0:
            if len(total_wins) > 25:
                win_pctg = np.sum(np.array(total_wins[-25:])) / len(total_wins[-25:])*100
            else:
                win_pctg = np.sum(np.array(total_wins)) / len(total_wins) * 100

            print('\nGame: ', i_episode,
                  '| Illegal moves: ', illegal_moves,
                  '| Loss: ', loss,
                  '| Win %:', str(win_pctg), '\n')

        total_illegal_moves.append(illegal_moves)

        if i_episode % 1000 == 0 and i_episode > 900:
            agent.PolicyNetwork.save_to_file()

    print('w: ', wins, ' l:', looses, ' t:', ties)
    agent.PolicyNetwork.save_to_file()

    fig = plt.figure()
    fig.canvas.set_window_title('Loss function across all episodes')
    fig.set_size_inches(11, 6)
    axs1 = fig.add_subplot(2, 2, 1)
    x = np.linspace(0, len(total_losses), len(total_losses))
    axs1.plot(x, total_losses, c='grey')
    moving_average_period = 64
    ma = np.convolve(total_losses, np.ones(moving_average_period), 'valid') / moving_average_period
    ma = np.concatenate((total_losses[:moving_average_period-1], ma))
    axs1.plot(x, ma, c='r')
    axs1.set_ylabel('Loss (squared error)', fontsize=10, color='.25')
    axs1.set_xlabel('Training Round', fontsize=10, color='.25')

    axs2 = fig.add_subplot(2, 2, 2)
    x2 = np.linspace(0, len(total_illegal_moves), len(total_illegal_moves))
    axs2.plot(x2, total_illegal_moves, c='grey')
    axs2.set_ylabel('Illegal Moves', fontsize=10, color='.25')
    axs2.set_xlabel('Episode', fontsize=10, color='.25')
    ma2 = np.convolve(total_illegal_moves, np.ones(moving_average_period), 'valid') / moving_average_period
    ma2 = np.concatenate((total_illegal_moves[:moving_average_period-1], ma2))
    axs2.plot(x2, ma2, c='r')

    axs3 = fig.add_subplot(2, 2, 3)
    x3 = np.linspace(0, len(total_wins), len(total_wins))
    ma3 = np.convolve(total_wins, np.ones(moving_average_period), 'valid') / moving_average_period * 100
    ma3 = np.concatenate((total_illegal_moves[:moving_average_period-1], ma3))
    axs3.set_ylabel('Win %', fontsize=10, color='.25')
    axs3.set_xlabel('Episode', fontsize=10, color='.25')
    axs3.plot(x3, ma3, c='r')
    plt.show()

# get N from input
c.N = int(input("Enter size of the board:"))
c.INPUTS = c.N * c.N * 3
c.OUTPUTS = c.N * c.N


if c.VISUAL:
    # initialize pygame and create window
    pyg.init()
    pyg.mixer.init()
    screen = pyg.display.set_mode((c.WIDTH, c.HEIGHT))
    pyg.display.set_caption("Tic Tac Toe")
    clock = pyg.time.Clock()
    all_sprites = pyg.sprite.Group()

    # load images and font
    images = [pyg.image.load('media/cross.png').convert(),
              pyg.image.load('media/circle.png').convert()]
    font_name = pyg.font.match_font('Calibri')

# initialize game
game = Game.Game()

# create neural network
experience = recordtype('experience', 'state action reward next_state')
agent = agent.Agent(c.INPUTS, c.HIDDEN_LAYERS, c.OUTPUTS, c.LEARNING_RATE)
replay_memory = Memory.ReplayMemory(c.MEMORY_CAPACITY)
agent.PolicyNetwork.load_from_file()
agent.TargetNetwork.copy_from(agent.PolicyNetwork)

if c.VISUAL:
    pygame_loop()

if not c.VISUAL and c.TRAIN:
    silent_training(game, agent, replay_memory)
elif not c.VISUAL and not c.TRAIN:
    play_wo_training(game, agent)

pyg.quit()
