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

        # This dictionary stores as keys the states that have already been
        # reached, i.e. the finally chosen states when calling the get_action
        # method. The values associated to keys correspond to the number of
        # times the given state has already been reached.
        self.statesAlreadyReached = dict()

        # The potential next reached state, i.e. the state corresponding to the
        # first movement of pacman and of the ghost in the expanded nodes.
        self.currentNextMoveState = None

        # The tuple representing the self.currentNextMoveState, i.e.
        # respectively the pacman position, the ghost position, and the food
        # matrix.
        self.currentNextMoveStateTuple = None

        # The number of calls to the get_action method.
        self.numberMoves = 0

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

        stateInfo = (
            state.getPacmanPosition(),
            state.getGhostPositions()[0],
            state.getFood())

        if stateInfo not in self.statesAlreadyReached:
            self.statesAlreadyReached[stateInfo] = 1

        else:
            self.statesAlreadyReached[stateInfo] += 1

        self.statesAlreadyPassedFromRoot = set()
        maxScore, maxMove = self.h_minimax_player_max(
            state, -(math.inf), math.inf, 0)

        self.numberMoves += 1

        return maxMove

    def h_minimax_player_max(
            self, currentState, maxValue, minValue, currentDepth):
        """
        This function computes the h-minimax value when the next move is done
        by Pacman (max player). The alpha-beta pruning is used.

        Arguments:
        ----------
        - 'currentState': The state to expand.

        - 'maxValue': The maximum h-minimax value computed by the max player at
                      the previous states of the expansion.

        - 'minValue': The minimum h-minimax value computed by the min player at
                      the previous states of the expansion.

        - 'currentDepth' : The depth of the recursion.

        Return:
        -------
        - 'maxScore': The maximum score found.

        - 'maxMove': The move corresponding to the maximum score found.

        """

        if self.cutoff_test(currentState, currentDepth):
            return self.score_evaluation(currentState), Directions.STOP

        # The two following instructions are used to not cycle indefinitely,
        # since the minimum score is unbounded.
        currentStateInfo1 = (
            currentState.getPacmanPosition(),
            currentState.getGhostPositions()[0],
            currentState.getFood(),
            True)
        self.statesAlreadyPassedFromRoot.add(currentStateInfo1)

        maxScore = -math.inf
        maxMove = Directions.STOP

        for successor in currentState.generatePacmanSuccessors():
            nextState = successor[0]
            nextMove = successor[1]

            # useful for the heuristic evaluation.
            if currentDepth == 1:
                self.currentNextMoveState = nextState
                self.currentNextMoveStateTuple = (
                    nextState.getPacmanPosition(),
                    nextState.getGhostPositions()[0],
                    nextState.getFood())

            nextStateInfo1 = (
                nextState.getPacmanPosition(),
                nextState.getGhostPositions()[0],
                nextState.getFood(),
                False)

            # If the state has already been passed in the path from the actual
            # state to the current state expanded, then we do not pass again on
            # it.
            if nextStateInfo1 not in self.statesAlreadyPassedFromRoot:
                resultScore, resultMove = self.h_minimax_player_min(
                    nextState, maxValue, minValue, currentDepth + 1)

            else:
                resultScore = math.inf

            if resultScore != math.inf and resultScore > maxScore:
                maxScore = resultScore
                maxMove = nextMove

                # If a predecessor node whose next move is done by min
                # player knows a path giving a smaller minimax value, then it
                # is useless to continue expanding the current state.
                if maxScore >= minValue:
                    self.statesAlreadyPassedFromRoot.remove(currentStateInfo1)
                    return maxScore, maxMove

                # The maximum value found is kept in order to inform the
                # successors of the biggest current minimax value.
                if maxValue < maxScore:
                    maxValue = maxScore

        # The passed state is removed in order to construct correctly the other
        # paths passing through other states.
        self.statesAlreadyPassedFromRoot.remove(currentStateInfo1)

        return maxScore, maxMove

    def h_minimax_player_min(
            self, currentState, maxValue, minValue, currentDepth):
        """
        This function computes the h-minimax value when the next move is done
        by the ghost (min player). The alpha-beta pruning is used.

        Arguments:
        ----------
        - 'currentState': The state to expand.

        - 'maxValue': The maximum h-minimax value computed by the max player at
                      the previous states of the expansion.

        - 'minValue': The minimum h-minimax value computed by the min player at
                      the previous states of the expansion.

        - 'currentDepth' : The depth of the recursion.

        Return:
        -------
        - 'minScore': The minimum score found.

        - 'minMove': The move corresponding to the minimum score found.

        """

        if self.cutoff_test(currentState, currentDepth):
            return self.score_evaluation(currentState), Directions.STOP

        # The two following instructions are used to not cycle indefinitely,
        # since the minimum score is unbounded.
        currentStateInfo1 = (
            currentState.getPacmanPosition(),
            currentState.getGhostPositions()[0],
            currentState.getFood(),
            False)
        self.statesAlreadyPassedFromRoot.add(currentStateInfo1)

        minScore = math.inf
        minMove = Directions.STOP

        for successor in currentState.generateGhostSuccessors(1):
            nextState = successor[0]
            nextMove = successor[1]

            # useful for the heuristic evaluation.
            if currentDepth == 1:
                self.currentNextMoveState = nextState
                self.currentNextMoveStateTuple = (
                    nextState.getPacmanPosition(),
                    nextState.getGhostPositions()[0],
                    nextState.getFood())

            nextStateInfo1 = (
                nextState.getPacmanPosition(),
                nextState.getGhostPositions()[0],
                nextState.getFood(),
                True)

            # If the state has already been passed in the path from the actual
            # state to the current state expanded, then we do not pass again on
            # it.
            if nextStateInfo1 not in self.statesAlreadyPassedFromRoot:
                resultScore, resultMove = self.h_minimax_player_max(
                    nextState, maxValue, minValue, currentDepth + 1)

            else:
                resultScore = (-math.inf)

            if resultScore != (-math.inf) and resultScore < minScore:
                minScore = resultScore
                minMove = nextMove

                # If a predecessor node whose next move is done by max
                # player knows a path giving a bigger minimax value, then it
                # is useless to continue expanding the current state.
                if minScore <= maxValue:
                    self.statesAlreadyPassedFromRoot.remove(currentStateInfo1)
                    return minScore, minMove

                # The minimum value found is kept in order to inform the
                # successors of the smallest current minimax value.
                if minValue > minScore:
                    minValue = minScore

        # The passed state is removed in order to construct correctly the other
        # paths passing through other states.
        self.statesAlreadyPassedFromRoot.remove(currentStateInfo1)

        return minScore, minMove

    def cutoff_test(self, state, depth):
        """
        This function is the cutoff evaluation function of the h-minimax
        algorithm.

        Arguments:
        ----------
        - 'state': The reached state with the h-minimax algorithm

        - 'depth': The current recursion depth

        Return:
        -------
        - True if the recursion must be stopped, False otherwise.
        """

        if state.isWin() or state.isLose():
            return True

        if depth >= 10:
            return True

        return False

    def score_evaluation(self, state):
        """
        This function computes the evaluated score corresponding to the
        potential of the state.

        Arguments:
        ----------
        - 'state': The state to be evaluated.

        Return:
        -------
        - The evaluated score.
        """

        if state.isWin():
            return state.getScore()

        return state.getScore() - self.find_total_penalties(state)

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

    def find_dots_positions(self, state):
        """
        This method returns a list of coordinates corresponding to the dots
        positions of the dots remaining on the layout.

        Arguments:
        ----------
        - 'state': The state whose dots positions must be computed.

        Return:
        -------
        - The list of dots positions of the dots remaining on the layout.
        """

        dotsPositions = []
        foodMatrix = state.getFood()
        i = 0
        while i < foodMatrix.width:
            j = 0
            while j < foodMatrix.height:
                if foodMatrix[i][j] == True:
                    dotsPositions.append((i, j))
                j += 1

            i += 1

        return dotsPositions

    def find_total_penalties(self, state):
        """
        This method computes the penalties linked to the given state.

        Arguments:
        ----------
        - 'state': The state whose penalties must be computed.

        Return:
        -------
        - The computed state penalties.
        """

        dosPosWithMD = []

        # The coordinates of all dots positions are obtained.
        dotsPositions = self.find_dots_positions(state)

        # The Manhattan distances between each dot position and the pacman
        # position are computed.
        for dotPosition in dotsPositions:
            manhattanDistance = self.manhattan_distance(
                state.getPacmanPosition(), dotPosition)
            dosPosWithMD.append((manhattanDistance, dotPosition[0], dotPosition[1]))

        # The values are sorted according to the Manhattan distances between
        # the dots positions and the pacman position.
        dosPosWithMD = sorted(dosPosWithMD)

        totalDistanceManhattan = 0
        i = 0

        # The first Manhattan distance is the distance between pacman and the
        # closest dot from pacman. This distance is multiplied by the number of
        # remaining dots on the layout to encourage getting the closest dot.
        totalDistanceManhattan += (dosPosWithMD[0][0] * len(dotsPositions))

        # The Manhattan distances between successives dots according to the
        # sorted values are computed and added to totalDistanceManhattan.
        # Manhattan distances are multiplied by the number of remaining
        # dots - 1 - i to encourage pacman getting the closest dots.
        while i < (len(dotsPositions) - 1):
            totalDistanceManhattan += (self.manhattan_distance((dosPosWithMD[i][1], dosPosWithMD[i][2]), (
                dosPosWithMD[i + 1][1], dosPosWithMD[i + 1][2])) * (len(dotsPositions) - 1 - i))
            i += 1

        # Additionnal penalties are put if the state has already been reached
        # before or if the state is a loosing state.
        if self.currentNextMoveStateTuple in self.statesAlreadyReached:
            if self.currentNextMoveState.isLose():
                return totalDistanceManhattan * len(dotsPositions) * (
                    self.statesAlreadyReached[self.currentNextMoveStateTuple] + 1) * (self.numberMoves + 1) + 1

            else:
                return totalDistanceManhattan * \
                    len(dotsPositions) * \
                    (self.statesAlreadyReached[self.currentNextMoveStateTuple] + 1)

        if self.currentNextMoveState.isLose():
            return totalDistanceManhattan * \
                len(dotsPositions) * (self.numberMoves + 1) + 1

        return totalDistanceManhattan * len(dotsPositions)
