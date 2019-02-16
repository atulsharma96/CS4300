# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")

        allFoodAsList = currentGameState.getFood().asList()
        currentPositionAsList = list(successorGameState.getPacmanPosition())
        distanceToReturn = float("-inf")

        if action == 'Stop':
            return distanceToReturn
        for currentGhostState in newGhostStates:
            if (currentGhostState.scaredTimer == 0) and currentGhostState.getPosition() == tuple(currentPositionAsList):
                return distanceToReturn

        for food in allFoodAsList:
            currentDistanceToFood = -1 * manhattanDistance(currentPositionAsList, food)
            if currentDistanceToFood > distanceToReturn:
                distanceToReturn = currentDistanceToFood

        return distanceToReturn

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.numOfAgents = gameState.getNumAgents()
        bestAction, score = self.minimax(gameState, self.depth, 0)
        return bestAction

    def minimax(self, gameState, depth, agentIndex):
        legalMoves = gameState.getLegalActions(agentIndex)
        if depth == 0 or len(legalMoves) == 0:
            return None, self.evaluationFunction(gameState)

        indexNextAgent = (agentIndex + 1) % self.numOfAgents  # Normalize the value of the index to stay in range
        depthNextAgent = depth

        if indexNextAgent == 0:
            depthNextAgent -= 1

        newAction = None

        if agentIndex == 0:
            valToReturn = float("-inf")
            for action in legalMoves:
                oldAction, newValue = self.minimax(gameState.generateSuccessor(agentIndex, action), depthNextAgent, indexNextAgent)
                valToReturn = max(newValue, valToReturn)
                if newValue == valToReturn:
                    newAction = action
            return newAction, valToReturn
        else:
            valToReturn = float("inf")
            for action in legalMoves:
                oldAction, newValue = self.minimax(gameState.generateSuccessor(agentIndex, action), depthNextAgent, indexNextAgent)
                valToReturn = min(newValue, valToReturn)
                if newValue == valToReturn:
                    newAction = action
            return newAction, valToReturn


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numOfAgents = gameState.getNumAgents()
        bestAction, score = self.alphabeta(gameState, self.depth, 0)
        return bestAction

    def alphabeta(self, gameState, depth, agentIndex, alpha=float("-inf"), beta=float("inf")):
        legalMoves = gameState.getLegalActions(agentIndex)
        if depth == 0 or len(legalMoves) == 0:
            return None, self.evaluationFunction(gameState)

        indexNextAgent = (agentIndex + 1) % self.numOfAgents  # Normalize the value of the index to stay in range
        depthNextAgent = depth

        if indexNextAgent == 0:
            depthNextAgent -= 1

        newAction = None

        if agentIndex == 0:
            valToReturn = float("-inf")
            for action in legalMoves:
                oldAction, newValue = self.alphabeta(gameState.generateSuccessor(agentIndex, action), depthNextAgent,
                                                     indexNextAgent, alpha, beta)
                valToReturn = max(newValue, valToReturn)
                if newValue == valToReturn:
                    newAction = action
                if valToReturn > beta:
                    break
                alpha = max(alpha, valToReturn)
        else:
            valToReturn = float("inf")
            for action in legalMoves:
                oldAction, newValue = self.alphabeta(gameState.generateSuccessor(agentIndex, action), depthNextAgent,
                                                     indexNextAgent, alpha, beta)
                valToReturn = min(newValue, valToReturn)
                if newValue == valToReturn:
                    newAction = action
                if valToReturn < alpha:
                    break
                beta = min(beta, valToReturn)
        return newAction, valToReturn


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.numOfAgents = gameState.getNumAgents()
        bestAction, score = self.expectimax(gameState, self.depth, 0)
        return bestAction

    def expectimax(self, gameState, depth, agentIndex):
        legalMoves = gameState.getLegalActions(agentIndex)
        if depth == 0 or len(legalMoves) == 0:
            return None, self.evaluationFunction(gameState)

        indexNextAgent = (agentIndex + 1) % self.numOfAgents  # Normalize the value of the index to stay in range
        depthNextAgent = depth

        if indexNextAgent == 0:
            depthNextAgent -= 1

        newAction = None

        if agentIndex == 0:
            valToReturn = float("-inf")
            for action in legalMoves:
                oldAction, newValue = self.expectimax(gameState.generateSuccessor(agentIndex, action), depthNextAgent,
                                                      indexNextAgent)
                valToReturn = max(newValue, valToReturn)
                if newValue == valToReturn:
                    newAction = action
            return newAction, valToReturn
        else:
            valToReturn = 0
            for action in legalMoves:
                oldAction, newValue = self.expectimax(gameState.generateSuccessor(agentIndex, action), depthNextAgent,
                                                      indexNextAgent)
                valToReturn += newValue
            valToReturn = valToReturn / float(len(legalMoves))
            return newAction, valToReturn


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The current point system incentivizes running away from
      ALL ghosts. We assign points based on the distance from the closest
      capsule and the closest food as well. We add 1 to act as a tie breaker
      in case pacman gets situated in an equidistant location from
      two food pellets.
    """
    "*** YOUR CODE HERE ***"

    CONSTANT_FOR_DECREMENTING_FACTOR = 5
    CONSTANT_FOR_INCREMENTING_FACTOR = 5
    CONSTANT_TO_CONSIDER_GHOSTS_MORE_SERIOUSLY = 3

    currentPosition = currentGameState.getPacmanPosition()
    allFoodAsList = currentGameState.getFood().asList()
    allCapsulesAsList = currentGameState.getCapsules()
    allGhostStates = currentGameState.getGhostStates()
    allScaredTimes = [ghostState.scaredTimer for ghostState in allGhostStates]
    score = currentGameState.getScore()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    # The ideal score should be
    # some tradeoff between food, ghosts
    # and booster capsules.

    closestFood = float("inf")
    closestCapsule = float("inf")
    closestGhost = float("inf")

    for food in allFoodAsList:
        distanceOfFood = manhattanDistance(food, currentPosition)
        if distanceOfFood < closestFood:
            closestFood = distanceOfFood

    # we do this decrement to always give food a high priority
    score -= closestFood

    for capsule in allCapsulesAsList:
        distanceOfCapsule = manhattanDistance(capsule, currentPosition)
        if distanceOfCapsule < closestCapsule:
            closestCapsule = distanceOfCapsule

    for ghost in allGhostStates:
        distanceOfGhost = manhattanDistance(ghost.getPosition(), currentPosition)
        if distanceOfGhost < closestGhost:
            closestGhost = distanceOfGhost

    if closestGhost < CONSTANT_TO_CONSIDER_GHOSTS_MORE_SERIOUSLY:
        if closestCapsule != 0 and closestGhost > closestCapsule:
            distanceToReturn = (score + 1 / (CONSTANT_FOR_DECREMENTING_FACTOR * closestCapsule))
        else:
            distanceToReturn = score + CONSTANT_FOR_INCREMENTING_FACTOR * closestGhost
    else:
        distanceToReturn = score

    return distanceToReturn

# Abbreviation
better = betterEvaluationFunction

