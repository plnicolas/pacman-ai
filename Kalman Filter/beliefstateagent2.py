from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'updateAndFetBeliefStates' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None
        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None
        # Uniform distribution size parameter 'w'
        # for sensor noise (see instructions)
        self.w = self.args.w
        # Probability for 'leftturn' ghost to take 'EAST' action
        # when 'EAST' is legal (see instructions)
        self.p = self.args.p
        self.i = 0
        
        
    def transitions(self, beliefStates, evidences):
        
        for i in range(0, len(evidences)):
            (width, height) = (beliefStates[i].shape[0], beliefStates[i].shape[1])
                
            #Predict step:
            #Element-wise multiplication of the current belief state and the transition matrix
            #Project the current belief state forward, from t to t+1, through the transition model
            beliefStates[i] = np.multiply(beliefStates[i], self.transitionMatrix(height, width))
            
            #Update step:
            #Element-wise multiplication of the predicted belief state and the sensor matrix
            #Update this new belief state using the fresh evidence
            beliefStates[i] = np.multiply(beliefStates[i], self.sensorMatrix(height, width, evidences[i]))

            #Normalize the matrix
            beliefStates[i] = self.normalize(beliefStates[i])

            
    def normalize(self, matrix):
        """
        Normalizes a matrix
        """
        sumOfElements = 0
        for i in range(0, matrix.shape[0]):    
            sumOfElements += sum(matrix[i])
        
        normalizedMatrix = matrix/sumOfElements
        return normalizedMatrix
        

    def sensorMatrix(self, height, width, ghostPos):
        """
        Creates the sensor matrix of a given ghost for the current time step
        """
        w2 = 2 * self.w + 1
        div = float(w2 * w2)
        sensorMatrix = np.ones((width, height))
        sensorMatrix *= 0.005
        (x, y) = ghostPos
        for i in range(x - self.w, x + self.w + 1):
            if(i >= 0 and i < width):
                for j in range(y - self.w, y + self.w + 1):
                    if(j >= 0 and j < height):
                        sensorMatrix[i][j] = (1.0/div)
        
        return sensorMatrix
            

    def _getLegalActions(self, wallMatrix, height, width, ghostPos):
        """
        Given a (ghost) position in the maze, returns the set of legal actions
        """
        walls = wallMatrix
        legalActions = []
        (x, y) = ghostPos
        if(x < width -1):
            if(walls[x+1][y] == 0):
                legalActions.append("East")
        if(x > 0):
            if(walls[x-1][y] == 0):
                legalActions.append("West")
        if(y < height - 1):
            if(walls[x][y+1] == 0):
                legalActions.append("North")
        if(y > 0):
            if(walls[x][y-1] == 0):
                legalActions.append("South")
    
        return legalActions


    def transitionMatrix(self, height, width):
        """
        Creates the transition matrix of the game
        """

        transMatrix = np.zeros((width, height))        
        for i in range(0, width):
            for j in range(0, height):
                    legalActions = self._getLegalActions(self.walls, height, width, (i,j))
                    n = len(legalActions)
                    #First, we determine if East is a legal action
                    if "East" in legalActions:
                        #East is legal; ghost selects it with probability p
                        transMatrix[i+1][j] += self.p
                        # or selects one of the others, with probability 1 - p
                        if "West" in legalActions:
                            transMatrix[i-1][j] += ((1 - self.p) / (n - 1))
                        if "North" in legalActions:
                            transMatrix[i][j+1] += ((1 - self.p) / (n - 1))
                        if "South" in legalActions:
                            transMatrix[i][j-1] += ((1 - self.p) / (n - 1))
                    else:
                        #East is not legal; ghost selects one of the legal actions,
                        #uniformly at random
                        if "West" in legalActions:
                            transMatrix[i-1][j] += (1/n)
                        if "North" in legalActions:
                            transMatrix[i][j+1] += (1/n)
                        if "South" in legalActions:
                            transMatrix[i][j-1] += (1/n)
                            
        #Probability of 0 where a wall is present
        for i in range(0, width):
            for j in range(0, height):
                if self.walls[i][j] == 1:
                    transMatrix[i][j] = 0

        transMatrix = self.normalize(transMatrix)
        
        return transMatrix

    def updateAndGetBeliefStates(self, evidences):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of (noised) ghost positions at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t} about ghost positions
          as N*M numpy matrices of probabilities
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """

        beliefStates = self.beliefGhostStates

        self.transitions(beliefStates, evidences)
        #util.pause()

        self.beliefGhostStates = beliefStates
        return beliefStates

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
        """
        positions = state.getGhostPositions()
        w = self.args.w
        w2 = 2*w+1
        div = float(w2 * w2)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w + 1):
                for j in range(y - w, y + w + 1):
                    dist[(i, j)] = 1.0 / div
            dist.normalize()
            new_positions.append(util.chooseFromDistribution(dist))
        return new_positions

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()
        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))
