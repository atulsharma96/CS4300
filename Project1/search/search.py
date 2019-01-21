# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    visited = list()
    frontier_stack = util.Stack()
    # to add some form of uniformity between the class and the implementation,
    # I have included the cost field in my tuple, but it is only ever used
    # by the UCS implementation.
    cost = 0.0
    frontier_stack.push((problem.getStartState(), list(), cost))
    while not frontier_stack.isEmpty():
        current_state, actions, cost_so_far = frontier_stack.pop()
        if problem.isGoalState(current_state):
            return actions
        for state, transition, cost in problem.getSuccessors(current_state):
            if state not in visited:
                visited.append(current_state)
                frontier_stack.push((state, actions + [transition], cost))


def breadthFirstSearch(problem):
    visited = list()
    willVisit = list()
    frontier_queue = util.Queue()
    # to add some form of uniformity between the class and the implementation,
    # I have included the cost field in my tuple, but it is only ever used
    # by the UCS implementation.
    cost = 0.0
    frontier_queue.push((problem.getStartState(), list(), cost))
    while not frontier_queue.isEmpty():
        current_state, actions, cost_so_far = frontier_queue.pop()
        if problem.isGoalState(current_state):
            return actions
        for state, transition, cost in problem.getSuccessors(current_state):
            if state not in visited and state not in willVisit:
                frontier_queue.push((state, actions + [transition], cost))
                willVisit.append(state)
        visited.append(current_state)



    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = list()
    frontier_queue = util.PriorityQueue()
    frontier_queue.push((problem.getStartState(), list()), 0.0)
    while not frontier_queue.isEmpty():
        current_state, actions = frontier_queue.pop()
        if current_state in visited:
            continue
        if problem.isGoalState(current_state):
            return actions
        for state, transition, cost in problem.getSuccessors(current_state):
            if state not in visited:
                frontier_queue.push((state, actions + [transition]), cost+problem.getCostOfActions(actions))
        visited.append(current_state)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = list()
    frontier_queue = util.PriorityQueue()
    frontier_queue.push((problem.getStartState(), list()), 0.0)
    while not frontier_queue.isEmpty():
        current_state, actions = frontier_queue.pop()
        if current_state in visited:
            continue
        if problem.isGoalState(current_state):
            return actions
        for state, transition, cost in problem.getSuccessors(current_state):
            cost += heuristic(state, problem)
            if state not in visited:
                frontier_queue.push((state, actions + [transition]), cost + problem.getCostOfActions(actions))
        visited.append(current_state)
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
