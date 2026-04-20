
class gameState:
    ACTIONS = ["move", "control"]
    #move, control, 

    def __init__(self, map_name, locs, intel = (0, 0)):
        if map_name == "four-cities":
            self.map = {1: (2, 3), 2: (1, 3), 3: (1, 2, 4), 4: (3)}
            self.major_cities = {1, 3}
        
        if player1_start not in self.map.keys:
            raise ValueError("Invalid starting city for player 1")
        if player2_start not in self.map.keys:
            raise ValueError("Invalid starting city for player 2")
        
        self.locs = [locs[0], locs[1]] #private
        self.ipt = [0, 0]
        self.intel [intel[0], intel[1]]
        self.controlled = [set(), set()]
        self.turn = 1
        self.public_actions = [[], []]

    def get_info_sets(self):
        current_player = self.turn % 2 + 1
    
    def is_terminal(self): #if is-terminal on the start of a turn, current player loses
        #

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
            if self.p1loc in self.major_cities:
                self.p1ipt += 4
            else:
                self.p1ipt += 1
        elif player == 2:
            self.p2control.append(self.p2loc)
            if self.p2loc in self.major_cities:
                self.p2ipt += 4
            else:
                self.p2ipt += 1
        else:
            raise ValueError("Invalid player number")
        
        