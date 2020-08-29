from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import Stack, Queue

from queue import PriorityQueue
from copy import deepcopy

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        # Used to store the coordinates of the dots.
        self.dotsPositions = []

        self.frontier = PriorityQueue()
        self.path = Queue()

    def manhattan_distance(self, position1, position2):
        """
        Given two pairs of coordinates, returns the Manhattan distance between
        the two positions.

        Arguments:
        ----------
        - 'position1': the first position
        - 'position2': the second position

        Return:
        -------
        - The Manhattan distance between the two positions.
        """

        distanceX = abs(position1[0] - position2[0])
        distanceY = abs(position1[1] - position2[1])
        return distanceX + distanceY

    def find_biggest_manhattan_distance(self, state):
        """
        This function computes the Manhattan distance between the position of
        the state given and all dots remaining. The maximum distance found is
        returned.

        Arguments:
        ----------
        - 'state': a state

        Return:
        -------
        - The maximum Manhattan distance found.
        """

        currentPosition = state.getPacmanPosition()

        foodMatrix = state.getFood()
        dotsRemaining = []

        for dotPosition in self.dotsPositions:
            if foodMatrix[dotPosition[0]][dotPosition[1]] == True:
                dotsRemaining.append(dotPosition)

        if dotsRemaining == []:
            return 0

        distanceMaximum = self.manhattan_distance(
            currentPosition, dotsRemaining[0])
        i = 1
        while i < len(dotsRemaining):
            if self.manhattan_distance(
                    currentPosition, dotsRemaining[i]) > distanceMaximum:
                distanceMaximum = self.manhattan_distance(
                    currentPosition, dotsRemaining[i])

            i += 1

        return distanceMaximum

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

        # If a path to a not-yet reached goal state has been found,
        # continue on that path until reached
        if not self.path.isEmpty():
            move = self.path.pop()
            return move

        # At the first execution, the dots positions must be searched.
        if self.dotsPositions == []:
            foodMatrix = state.getFood()
            i = 0
            while i < foodMatrix.width:
                j = 0
                while j < foodMatrix.height:
                    if foodMatrix[i][j] == True:
                        self.dotsPositions.append((i, j))
                    j += 1

                i += 1

        # If there is no dots, pacman does not move.
        if self.dotsPositions == []:
            return Directions.STOP

        self.frontier = PriorityQueue()

        pathInformation = PathInformation(
            0, self.find_biggest_manhattan_distance(state), state, [])

        self.frontier.put_nowait(pathInformation)

        # A state is defined by a (x,y) position in the labyrinth, and the remaining number of dots
        # along with their position, i.e. the food matrix
        self.visited = {(state.getPacmanPosition(), state.getFood())}

        # While the frontier is not empty, remove one state from it and explore
        # further
        while not self.frontier.empty():

            queueHead = self.frontier.get_nowait()
            currentBackwardCost = queueHead.backwardCost
            currentState = queueHead.state
            currentMoves = queueHead.moves

            # If the removed state is a goal state, return the path leading to
            # it
            if currentState.isWin():
                self._build_path(currentMoves)
                nextMove = self.path.pop()
                return nextMove

            # Else, expand the frontier with the successors of the current
            # state
            self._explore_frontier(
                currentState,
                currentBackwardCost,
                currentMoves)

        return Directions.STOP

    def _explore_frontier(self, state, currentBackwardCost, currentMoves):
        """
        Given a pacman game state,the path leading to it (along with its cost),
        expand the frontier by adding its not-yet-visited successors.

        Arguments:
        ----------
        - `state`: the current game state.
        - `currentBackwardCost`: the cost to reach this state
        - `currentMoves`: the path to reach this state
        """

        currentFood = state.getFood()

        # Generate the successors of the current state
        successors = state.generatePacmanSuccessors()
        for s in successors:

            nextState = s[0]
            nextMove = s[1]

            # The path to the successor is the path to the current state plus
            # the action to go from the current state to its successor
            nextMoves = deepcopy(currentMoves)
            nextMoves.append(nextMove)

            # If we move unto a cell with food, it costs "less" than if we move
            # unto a cell without food (since eating increases our score)
            nextPacmanPosition = nextState.getPacmanPosition()
            if currentFood[nextPacmanPosition[0]
                           ][nextPacmanPosition[1]] == True:
                costIncrease = 1
            else:
                costIncrease = 11

            sInfo = (s[0].getPacmanPosition(), s[0].getFood())

            # If the successor has not been visited yet, add it to the frontier,
            # and mark it as visited.
            if sInfo not in self.visited:
                nextPathInformation = PathInformation(currentBackwardCost + costIncrease,
                                                      self.find_biggest_manhattan_distance(
                                                          nextState),
                                                      nextState, nextMoves)
                self.frontier.put_nowait(nextPathInformation)
                self.visited.add(sInfo)

    def _build_path(self, statePath):
        """
        Given a list of directions leading to a certain state,
        build the path as a queue to be used by the get_action method.

        Arguments:
        ----------
        - `statePath`: a path, stored as a list of directions.
        """
        i = 0
        while i <= (len(statePath) - 1):
            self.path.push(statePath[i])
            i += 1


class PathInformation:
    """
    This class is used to store useful information about a developped path.
    """

    def __init__(self, backwardCost, forwardCost, state, moves):
        """
        Initialisation function.

        Arguments:
        ----------
        - backwardCost: The backward cost.
        - forwardCost: The forward cost.
        - state : The state reached by the path.
        - moves: The moves to make from the current state to reach the state
                 stored in this object.
        """
        self.backwardCost = backwardCost
        self.forwardCost = forwardCost
        self.state = state
        self.moves = moves

    def __lt__(self, otherPathInformation):
        """
        This method compares this PathInformation object with another one and
        return True if this object is strictly lower than otherPathInformation
        object.

        Arguments:
        ----------
        - otherPathInformation: a PathInformation object to compare with this
                                PathInformation object.
        Return:
        -------
        - True if self.cost < otherPathInformation.cost, False otherwise.
        """
        return (self.backwardCost + self.forwardCost) < (
            otherPathInformation.backwardCost + otherPathInformation.forwardCost)
