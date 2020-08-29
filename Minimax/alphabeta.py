from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import Stack

import math

class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        # This variable corresponds to a set (instanciated at each call of the
        # get_action method) that stores the states that have already been
        # passed in the path from the actual state to the expanded state.
        self.statesAlreadyPassedFromRoot = None

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

        self.statesAlreadyPassedFromRoot = set()
        maxScore, maxMove = self.minimax_player_max(
            state, -(math.inf), math.inf)

        return maxMove

    def minimax_player_max(self, currentState, maxValue, minValue):
        """
        This function computes the minimax value when the next move is done by
        Pacman (max player). The alpha-beta pruning is used.

        Arguments:
        ----------
        - 'currentState': The state to expand.

        - 'maxValue': The maximum minimax value computed by the max player at
                      the previous states of the expansion.

        - 'minValue': The minimum minimax value computed by the min player at
                      the previous states of the expansion.


        Return:
        -------
        - 'maxScore': The maximum score found.

        - 'maxMove': The move corresponding to the maximum score found.
        """

        if currentState.isWin() or currentState.isLose():
            return currentState.getScore(), Directions.STOP

        # The two following instructions are used to not cycle indefinitely,
        # since the minimum score is unbounded.
        currentStateInfo = (
            currentState.getPacmanPosition(),
            currentState.getGhostPositions()[0],
            currentState.getFood(),
            True)
        self.statesAlreadyPassedFromRoot.add(currentStateInfo)

        maxScore = -math.inf
        maxMove = Directions.STOP

        for successor in currentState.generatePacmanSuccessors():
            nextState = successor[0]
            nextMove = successor[1]

            nextStateInfo = (
                nextState.getPacmanPosition(),
                nextState.getGhostPositions()[0],
                nextState.getFood(),
                False)

            # If the state has already been passed in the path from the actual
            # state to the current state expanded, then we do not pass again on
            # it.
            if nextStateInfo not in self.statesAlreadyPassedFromRoot:
                resultScore, resultMove = self.minimax_player_min(
                    nextState, maxValue, minValue)

            else:
                resultScore = math.inf

            if resultScore != math.inf and resultScore > maxScore:
                maxScore = resultScore
                maxMove = nextMove

                # If a predecessor node whose next move is done by min
                # player knows a path giving a smaller minimax value, then it
                # is useless to continue expanding the current state.
                if maxScore >= minValue:
                    self.statesAlreadyPassedFromRoot.remove(currentStateInfo)
                    return maxScore, maxMove

                # The maximum value found is kept in order to inform the
                # successors of the biggest current minimax value.
                if maxValue < maxScore:
                    maxValue = maxScore

        # The passed state is removed in order to construct correctly the other
        # paths passing through other states.
        self.statesAlreadyPassedFromRoot.remove(currentStateInfo)

        return maxScore, maxMove

    def minimax_player_min(self, currentState, maxValue, minValue):
        """
        This function computes the minimax value when the next move is done by
        the ghost (min player). The alpha-beta pruning is used.

        Arguments:
        ----------
        - 'currentState': The state to expand.

        - 'maxValue': The maximum minimax value computed by the max player at
                      the previous states of the expansion.

        - 'minValue': The minimum minimax value computed by the min player at
                      the previous states of the expansion.

        Return:
        -------
        - 'minScore': The minimum score found.

        - 'minMove': The move corresponding to the minimum score found.
        """

        if currentState.isWin() or currentState.isLose():
            return currentState.getScore(), Directions.STOP

        # The two following instructions are used to not cycle indefinitely,
        # since the minimum score is unbounded.
        currentStateInfo = (
            currentState.getPacmanPosition(),
            currentState.getGhostPositions()[0],
            currentState.getFood(),
            False)
        self.statesAlreadyPassedFromRoot.add(currentStateInfo)

        minScore = math.inf
        minMove = Directions.STOP

        for successor in currentState.generateGhostSuccessors(1):
            nextState = successor[0]
            nextMove = successor[1]

            nextStateInfo = (
                nextState.getPacmanPosition(),
                nextState.getGhostPositions()[0],
                nextState.getFood(),
                True)

            # If the state has already been passed in the path from the actual
            # state to the current state expanded, then we do not pass again on
            # it.
            if nextStateInfo not in self.statesAlreadyPassedFromRoot:
                resultScore, resultMove = self.minimax_player_max(
                    nextState, maxValue, minValue)

            else:
                resultScore = (-math.inf)

            if resultScore != (-math.inf) and resultScore < minScore:
                minScore = resultScore
                minMove = nextMove

                # If a predecessor node whose next move is done by max
                # player knows a path giving a bigger minimax value, then it
                # is useless to continue expanding the current state.
                if minScore <= maxValue:
                    self.statesAlreadyPassedFromRoot.remove(currentStateInfo)
                    return minScore, minMove

                # The minimum value found is kept in order to inform the
                # successors of the smallest current minimax value.
                if minValue > minScore:
                    minValue = minScore

        # The passed state is removed in order to construct correctly the other
        # paths passing through other states.
        self.statesAlreadyPassedFromRoot.remove(currentStateInfo)

        return minScore, minMove
