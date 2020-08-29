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

        # The transition matrix
        self.transitionMatrix = []

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
        # XXX: Your code here

        width = beliefStates[0].shape[0]
        height = beliefStates[0].shape[1]

        i = 0

        # The list contains the beliefStates for all ghosts in the maze.
        listBeliefStates = list()
        for evidence in evidences:
            xPositionGhost = evidence[0]
            yPositionGhost = evidence[1]

            newBeliefStates = self.compute_probabilities(
                width, height, xPositionGhost, yPositionGhost, beliefStates[i])
            listBeliefStates.append(newBeliefStates)

            i += 1

        beliefStates = listBeliefStates

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def _getLegalActions(self, width, height, x, y):
        """
        Given a (ghost) position in the maze together with the width and
        height of the maze, returns the list of legal actions.

        Arguments:
        ----------
        - width: The width of the maze.
        - height: The height of the maze.
        - x: The abscissa position of the ghost.
        - y: The ordinate position of the ghost.

        Return:
        -------
        - The list of legal actions given the ghost position in the maze.

        """

        walls = self.walls
        legalActions = []
        if(x < width - 1):
            if(walls[x + 1][y] == False):
                legalActions.append("East")
        if(x > 0):
            if(walls[x - 1][y] == False):
                legalActions.append("West")
        if(y < height - 1):
            if(walls[x][y + 1] == False):
                legalActions.append("North")
        if(y > 0):
            if(walls[x][y - 1] == False):
                legalActions.append("South")

        return legalActions

    def create_transition_matrix(self, width, height):
        """
        This method creates the transition matrix of the maze.

        Arguments:
        ----------
        - width: The width of the maze.
        - height: The height of the maze.

        Return:
        -------
        - The transition matrix of the maze.
        """

        self.transitionMatrix = np.zeros((width * height, width * height))

        i = 0
        while i < (width * height):
            xPositionInitial = int(i / height)
            yPositionInitial = int(i % height)

            # In a transition matrix, the sum of the rows must be equal to 1.
            # If we are on a wall, we stay in the same place.
            if self.walls[xPositionInitial][yPositionInitial]:

                position = xPositionInitial * height + yPositionInitial
                self.transitionMatrix[i, position] = 1.0

                i += 1
                continue

            legalActions = self._getLegalActions(
                width, height, xPositionInitial, yPositionInitial)

            # If we are blocked surrounded by walls (like pacman in this
            # case), we cannot move.
            if legalActions == []:
                position = xPositionInitial * height + yPositionInitial
                self.transitionMatrix[i, position] = 1.0

            if "East" in legalActions:
                position = (xPositionInitial + 1) * height + yPositionInitial
                self.transitionMatrix[i, position] = self.p + \
                    (1 - self.p) * (1.0 / len(legalActions))

                if "West" in legalActions:
                    position = (xPositionInitial - 1) * \
                        height + yPositionInitial
                    self.transitionMatrix[i, position] = (
                        1 - self.p) * (1.0 / len(legalActions))

                if "North" in legalActions:
                    position = xPositionInitial * \
                        height + (yPositionInitial + 1)
                    self.transitionMatrix[i, position] = (
                        1 - self.p) * (1.0 / len(legalActions))

                if "South" in legalActions:
                    position = xPositionInitial * \
                        height + (yPositionInitial - 1)
                    self.transitionMatrix[i, position] = (
                        1 - self.p) * (1.0 / len(legalActions))

            else:

                if "West" in legalActions:
                    position = (xPositionInitial - 1) * \
                        height + yPositionInitial
                    self.transitionMatrix[i,
                                          position] = 1.0 / len(legalActions)

                if "North" in legalActions:
                    position = xPositionInitial * \
                        height + (yPositionInitial + 1)
                    self.transitionMatrix[i,
                                          position] = 1.0 / len(legalActions)

                if "South" in legalActions:
                    position = xPositionInitial * \
                        height + (yPositionInitial - 1)
                    self.transitionMatrix[i,
                                          position] = 1.0 / len(legalActions)

            i += 1

        return self.transitionMatrix

    def create_observation_matrix(
            self,
            width,
            height,
            xPositionGhost,
            yPositionGhost):
        """
        This function creates the observation matrix centered in
        (xPositionGhost, yPositionGhost) for the sensor model.

        Arguments:
        ----------
        - width: The width of the maze.
        - height: The height of the maze.
        - xPositionGhost: The abcissa of the noised position of the ghost.
        - yPositionGhost: The ordinate of the noised position of the ghost.

        Return:
        -------
        - The observation matrix for the sensor model.
        """

        w = self.w
        w2 = 2 * w + 1
        div = w2 * w2
        matrixValues = np.zeros((width, height))

        # used for normalizing to 1.
        totalSum = 0.0

        # The sensor model follows a uniform discrete distribution.
        for i in range(xPositionGhost - w, xPositionGhost + w + 1):
            for j in range(yPositionGhost - w, yPositionGhost + w + 1):
                if (i >= 0) and (i < width) and (j >= 0) and (j < height):
                    matrixValues[i, j] = 1.0 / div
                    totalSum += matrixValues[i, j]

        # The values are normalized to 1.
        for i in range(xPositionGhost - w, xPositionGhost + w + 1):
            for j in range(yPositionGhost - w, yPositionGhost + w + 1):
                if (i >= 0) and (i < width) and (j >= 0) and (j < height):
                    matrixValues[i, j] /= totalSum

        vectorValues = matrixValues.flatten()
        observationMatrix = np.diag(vectorValues)

        return observationMatrix

    def compute_probabilities(
            self,
            width,
            height,
            xPositionGhost,
            yPositionGhost,
            beliefStates):
        """
        This function computes the belief states given the belief states of the previous time.

        Arguments:
        ----------
        - width: The width of the maze.
        - height: The height of the maze.
        - xPositionGhost: The noised abscissa position of the ghost.
        - yPositionGhost: The noised ordinate position of the ghost.
        - beliefStates: The belief states for the previous time.

        Return:
        -------
        - The belief states of the current time.
        """

        # The transition matrix is the same for all steps and is thus computed
        # only once.
        if self.transitionMatrix == []:
            self.transitionMatrix = self.create_transition_matrix(
                width, height)

        observationMatrix = self.create_observation_matrix(
            width, height, xPositionGhost, yPositionGhost)

        beliefStates = np.array(beliefStates)
        beliefStatesVector = beliefStates.flatten()
        beliefStatesVector = beliefStatesVector.reshape(-1, 1)

        resultVector = np.dot(
            observationMatrix,
            np.dot(
                self.transitionMatrix.T,
                beliefStatesVector))

        resultVector = resultVector.reshape(-1)

        sumElements = np.sum(resultVector)

        i = 0

        # The vector corresponding to the belief states is normalized to 1.
        while i < len(resultVector) and sumElements != 0.0:
            resultVector[i] /= sumElements
            i += 1

        newBeliefStates = resultVector.reshape(width, height)

        return newBeliefStates

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
        """
        positions = state.getGhostPositions()
        w = self.args.w
        w2 = 2 * w + 1
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
