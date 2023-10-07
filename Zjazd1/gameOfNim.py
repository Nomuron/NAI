"""Game of Nim project for AI tools course
Authors: Magdalena Asmus-Mrzygłód, Patryk Klimek

In order to be able to run script with this game you will need:
Python at least 3.8
easyAI library at least 2.0
Link to install python: https://www.python.org/downloads/
When you have installed python in the command line run command 'pip install easyAI'
When your library is installed, in command line move to folder where you have this file
and run command 'python gameOfNim.py'
"""
from easyAI import TwoPlayerGame, Negamax, Human_Player, AI_Player


class GameOfNim(TwoPlayerGame):
    """There is at least a few variations of game of Nim, this is a ruleset which we decided to code.
    Game of Nim is a game similar to game of bones. You have 3 heaps with tokens.
    Every round one player can take up to 3 tokens from only one heap, then his turn ends.
    The winner will be a player which will take the last token or tokens"""

    def __init__(self, players=None):
        self.players = players
        self.pile = [7, 8, 9]  # start with 3 heaps of tokens, first one has 7 tokens, second has 8, third has 9
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        """Generating of list of possible moves

        Returns:
        List[str]: List of possible moves to be done
        """
        moves = []
        for i in range(3):
            for j in range(1, 4):
                if self.pile[i] >= j:  # checking if pile big enough for specific move
                    moves.append("{},{}".format(i + 1, j))
        return moves

    def make_move(self, move):
        """Executing moves chosen by players

        Parameters:
        move (str): specific move chosen by player to be done
        """
        move_details = move.split(",")  # split moves by coma
        if self.pile[int(move_details[0]) - 1] > 0:  # check if there are still tokens on a heap
            self.pile[int(move_details[0]) - 1] -= (int(move_details[1]))  # remove tokens from heap

    def win(self):
        """Function that define rules of the winning of the game

        Returns:
        bool: returns true if winning requirements are met
        """
        if self.pile.count(0) == 3:  # game ends when last player will take last token
            return True

    def is_over(self):
        """Functions which stop the game when win function return true

        Returns:
        bool: return results of winning from win function
        """
        return self.win()  # game stops when someone wins.

    def scoring(self):
        """Function that return score to player who will win a game

        Returns:
        int: chosen score number of points which player receive when they win
        """
        return 100 if self.win() else 0

    def show(self):
        """Function to print amount of tokens on specific heaps"""
        print('First heap contains {} tokens \nSecond heap contains {} tokens \nThird heap contains {} tokens'
              .format(self.pile[0], self.pile[1], self.pile[2]))


ai = Negamax(10)  # The AI will think 10 moves in advance
game = GameOfNim([AI_Player(ai), AI_Player(ai)])  # run a game with human player and AI player
history = game.play()


happy_monkey = """                @&&&&&&&&&&&&%@                          .    
                            @&&&&&&&&&&&&&&&&&&&&&&&&%/                         
                         %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%@                      
                       %&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@                    
                     (&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%                   
                    @&&&&&&&%%%%&&%&&&&&&&&&&&&%%&@@%%&&&&&&&&@                 
            @&&&&&&%&&&&%&.,,,,,,,,,,.,(%&%&@,,,,,,,,,,,,.%&&&&@&&&&&&&         
         %&&&&%&&%&%&&&%&,,,,,,,,,,,.,,,, .,,,,,,,,,,,,,,,,.&&&%%%%%&&&&&&      
       @&&&&@..,,,.&&&&@,,,,,,,,     .,,,,,,..      ...,,,,,&&&%&,,,,,,&%%&&    
       &&&%@,,,,,,,&&&%.,,,,,,%       @,,,,,.        .,,,,,,.&&&%,,,,,,.&&&&@   
      &&&&&&,,,,,,,&&&&.,,,,,   @@@@@  ,,,,,(  .@@,  @,,,,,, &&&&,,,,,,@&&&&&   
       %%&&&&,,,,,,&&&%.,,,,,. /%@@@@    .. @ #&@@@/ @,,,,,,@&&%@,,,,,&&&&&%    
        .&&&&&%&,,,%&%%@,,,,,,@ .@@@, (,.....  #@@@@,,,,,,, @%&%,,, &&&&&&&     
           @&&&&&%@,,,,,,,,,,.@..,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.@%&&&&&.       
               @,,,,,,,,,,,,,,,,,,,,,,@@     @@,,,,,,,,,,,,,,,,,,,,,            
               .,,..,,,,**///***,,,,,,,@@@@@@@,,,,,,,,**//****,,,,,,,       .   
               .,,,,,,&&&&&&&///*,.*,,,,,,,,,,,,,,,,*/////&&&&&/*,,,,,,      .    
                ,,,,,,,,*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*///**,.@           
                 %,,,,,,,,,,,,,&&&&...............&&&&&&&,,,,,,,,,,,,,.             
                   (,,,,,,,,,,,,,,,.&&&&&&&&&&&&&,,,,,,,,,,,,,.@               
                       @,,,,,,,,,,,,,,,&&&&&&&,,,,,,,,,,,.,,@                   
                             @&.,,,,,,,,,,,,,,,,,,..,*@    """

sad_monkey = """                  @&&&&&&&&&&&&%@                          .    
                            @&&&&&&&&&&&&&&&&&&&&&&&&%/                         
                         %&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%@                      
                       %&%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&@                    
                     (&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%                   
                    @&&&&&&&%%%%&&%&&&&&&&&&&&&%%&@@%%&&&&&&&&@                 
            @&&&&&&%&&&&%&.,,,,,,,,,,.,(%&%&@,,,,,,,,,,,,.%&&&&@&&&&&&&         
         %&&&&%&&%&%&&&%&,,,,,,,,,,,.,,,, .,,,,,,,,,,,,,,,,.&&&%%%%%&&&&&&      
       @&&&&@..,,,.&&&&@,,,,,,,,     .,,,,,,..      ...,,,,,&&&%&,,,,,,&%%&&    
       &&&%@,,,,,,,&&&%.,,,,,,%       @,,,,,.        .,,,,,,.&&&%,,,,,,.&&&&@   
      &&&&&&,,,,,,,&&&&.,,,,,   @@@@@  ,,,,,(  .@@,  @,,,,,, &&&&,,,,,,@&&&&&   
       %%&&&&,,,,,,&&&%.,,,,,. /%@@@@    .. @ #&@@@/ @,,,,,,@&&%@,,,,,&&&&&%    
        .&&&&&%&,,,%&%%@,,,,,,@ .@@@, (,.....  #@@@@,,,,,,, @%&%,,, &&&&&&&     
           @&&&&&%@,,,,,,,,,,.@..,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.@%&&&&&.       
               @,,,,,,,,,,,,,,,,,,,,,,@@     @@,,,,,,,,,,,,,,,,,,,,,            
               .,,..,,,,**///***,,,,,,,@@@@@@@,,,,,,,,**//****,,,,,,,       .   
               .,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,      .    
                ,,,,,,,,,,,,,,,,,,,&&&&&&&&&&&&&&&,,,,,,,,,,,,,,,,.@           
                 %,,,,,,,,,,,,&&&&&&&&&&&&&&&&&&&&&&&&&&,,,,,,,,,,,,,.             
                   (,,,,,,,,,&&&&&&&&&&&&&&&&&&&&&&&&&&&&&,,,,,.@               
                       @,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.,,@                   
                             @&.,,,,,,,,,,,,,,,,,,..,*@                        """
