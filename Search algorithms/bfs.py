from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import Stack, Queue

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        self.frontier = Queue()
        self.path = Queue()

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
        # continue on that path until the goal is reached
        if not self.path.isEmpty():
            move = self.path.pop()
            return move

        # Breadth-First-Search: Frontier is a FIFO queue
        self.frontier = Queue()

        # A state is defined by a (x,y) position in the labyrinth, and the remaining number of dots
        # along with their position, i.e. the food matrix
        # self.visited = [[state.getPacmanPosition(), state.getFood()]]
        self.visited = {(state.getPacmanPosition(), state.getFood())}

        self._explore_frontier(state, [])

        # While the frontier is not empty, remove one state from it and explore
        # further
        while not self.frontier.isEmpty():

            [currentState, statePath] = self.frontier.pop()

            # If the removed state is a goal state, return the path leading to
            # it
            if currentState.isWin():
                self._build_path(statePath)
                nextMove = self.path.pop()
                return nextMove

            # Else, expand the frontier with the successors of the current
            # state
            self._explore_frontier(currentState, statePath)

        return Directions.STOP

    def _explore_frontier(self, state, statePath):
        """
        Given a pacman game state and the path leading to it,
        expand the frontier by adding its not-yet-visited successors.

        Arguments:
        ----------
        - `state`: the current game state.
        - `statePath`: the path leading to the current state.
        """

        # Generate the successors of the current state
        successors = state.generatePacmanSuccessors()
        for s in successors:
            # The path to the successor is the path to the current state plus
            # the action to go from the current state to its successor
            sPath = statePath + [s[1]]
            # sInfo = [s[0].getPacmanPosition(), s[0].getFood()]
            sInfo = (s[0].getPacmanPosition(), s[0].getFood())

            # If the successor has not been visited yet, add it to the frontier,
            # and mark it as visited.
            if sInfo not in self.visited:
                self.frontier.push([s[0], sPath])
                # self.visited.append(sInfo)
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
