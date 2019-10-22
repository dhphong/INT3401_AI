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
        newGhostStates = successorGameState.getGhostStates()

        pacman_food = [manhattanDistance(newPos, food) for food in currentGameState.getFood().asList()]

        if action == 'Stop':
            return float('-Inf')
        for state in newGhostStates:
            if state.getPosition() == newPos and (state.scaredTimer == 0):
                return float("-Inf")
        score = -1 * min(pacman_food)
        return score

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
        def maxValue(agentIndex, depth, gameState):
            childValue = [minimax(agentIndex + 1, depth, gameState.generateSuccessor(agentIndex, state)) for state in gameState.getLegalActions(agentIndex)]
            return max(childValue)

        def minValue(agentIndex, depth, gameState):
            nextAgent = agentIndex + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            childValue = [minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, state)) for state in gameState.getLegalActions(agentIndex)]
            return min(childValue)

        def minimax(agentIndex, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            else:
                return minValue(agentIndex, depth, gameState)

        maximum = float('-Inf')
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            temp = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if temp > maximum or maximum == float('-inf'):
                maximum = temp
                action = agentState

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxValue(agentIndex, depth, gameState, a, b):
            v = float('-inf')
            for state in gameState.getLegalActions(agentIndex):
                v = max(v, alphabeta(1, depth, gameState.generateSuccessor(agentIndex, state), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def minValue(agentIndex, depth, gameState, a, b):
            v = float('inf')
            nextAgent = agentIndex + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1

            for state in gameState.getLegalActions(agentIndex):
                v = min(v, alphabeta(nextAgent, depth, gameState.generateSuccessor(agentIndex, state), a, b))
                if v < a:
                    return v
                b = min(v, b)
            return v

        def alphabeta(agentIndex, depth, gameState, a, b):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, a, b)
            else:
                return minValue(agentIndex, depth, gameState, a, b)

        maximum = float('-Inf')
        action = Directions.WEST
        alpha = float('-inf')
        beta = float('inf')
        for agentState in gameState.getLegalActions(0):
            temp = alphabeta(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if temp > maximum or maximum == float('-inf'):
                maximum = temp
                action = agentState
            if maximum > beta:
                return maximum
            alpha = max(alpha, maximum)
        return action

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

        def maxValue(agentIndex, depth, gameState):
            childValue = [expectimax(1, depth, gameState.generateSuccessor(agentIndex, state)) for state in gameState.getLegalActions(agentIndex)]
            return max(childValue)

        def expValue(agentIndex, depth, gameState):
            nextAgent = agentIndex + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:
                depth += 1
            childValue = [expectimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, state)) for state in gameState.getLegalActions(agentIndex)]
            return sum(childValue) / float(len(childValue))

        def expectimax(agentIndex, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            else:
                return expValue(agentIndex, depth, gameState)

        maximum = float('-inf')
        action = Directions.WEST
        for state in gameState.getLegalActions(0):
            temp = expectimax(1, 0, gameState.generateSuccessor(0, state))
            if temp > maximum or maximum == float('-inf'):
                maximum = temp
                action = state
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    foodDistance = [util.manhattanDistance(newPos, food) for food in newFood]
    min_food_distance = min(foodDistance) if len(foodDistance) > 0 else -1

    ghostDistance = [util.manhattanDistance(newPos, ghost) for ghost in currentGameState.getGhostPositions()]
    distances_to_ghosts = 1 + sum(ghostDistance)
    proximity_to_ghosts = len([ghost for ghost in ghostDistance if ghost <= 1])

    newCapsule = currentGameState.getCapsules()
    numberOfCapsules = len(newCapsule)

    return currentGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts - numberOfCapsules

# Abbreviation
better = betterEvaluationFunction

