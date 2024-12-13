
class gameMap:
    def __init__(self, map_name, player1_start, player2_start):
        if map_name == "four-cities":
            self.map = {1: (2, 3), 2: (1, 3), 3: (1, 2, 4), 4: (3)}
        
        if player1_start not in self.map.keys:
            raise ValueError("Invalid starting city for player 1")

        if player2_start not in self.map.keys:
            raise ValueError("Invalid starting city for player 2")
        
        self.p1loc = player1_start
        self.p2loc = player2_start

        self.p1control = set()
        self.p2control = set()

    def move(self, player, destination):
        if player == 1:
            if destination not in self.map[self.p1loc]:
                raise ValueError("Invalid move for player 1")
            self.p1loc = destination
        elif player == 2:
            if destination not in self.map[self.p2loc]:
                raise ValueError("Invalid move for player 2")
            self.p2loc = destination
        else:
            raise ValueError("Invalid player number")
        
    def control(self, player):
        if player == 1:
            self.p1control.append(self.p1loc)
        elif player == 2:
            self.p2control.append(self.p2loc)
        else:
            raise ValueError("Invalid player number")
        
    def 


        





class gameState:
    def __init__():
        self.map = map
