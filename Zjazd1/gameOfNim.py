""" This is the example featured in section 'A quick example' of the docs """

from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player
import numpy as np


class GameOfNim(TwoPlayerGame):
    """In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses."""

    def __init__(self, players=None):
        self.players = players
        self.pile = [7, 8, 9]  # start with 3 heaps of tokens, first one has 7 tokens, second has 8, third has 9
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        return ["1,1", "1,2", "1,3", "2,1", "2,2", "2,3", "3,1", "3,2", "3,3"]

    def make_move(self, move):
        move_details = move.split(",")
        if self.pile[int(move_details[0]) - 1] > 0:
            self.pile[int(move_details[0]) - 1] -= (int(move_details[1]))  # remove bones.

    def win(self):
        if self.pile.count(0) == 2 and self.pile[np.nonzero(self.pile)[0][0]] < 4:
            return True   # opponent took the last bone ?

    def is_over(self):
        return self.win()  # game stops when someone wins.

    def scoring(self):
        return 100 if self.win() else 0

    def show(self):
        print('First heap contains {} tokens \nSecond heap contains {} tokens \nThird heap contains {} tokens'
              .format(self.pile[0], self.pile[1], self.pile[2]))


ai = Negamax(10)  # The AI will think 10 moves in advance
game = GameOfNim([Human_Player(), AI_Player(ai)])
history = game.play()
