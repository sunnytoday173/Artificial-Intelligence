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
    return  [s, s, w, s, w, w, s, w]

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
    from util import Stack
    from game import Directions
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self,initial_state,parent=None,action=None):
            self.state=initial_state
            self.parent=parent
            self.action=action
    node = Node(problem.getStartState())
    if problem.isGoalState(node.state):
       return solution(node)
    frontier = Stack()
    frontier.push(node)
    explored=set()
    while True:
        node = frontier.pop()
        if node.state not in explored:
            if problem.isGoalState(node.state):
                return solution(node)
            explored.add(node.state)
            for child in problem.getSuccessors(node.state):
                childnode = Node(child[0], node, child[1])
                if (childnode.state not in explored) and (childnode not in frontier.list):
                    frontier.push(childnode)
        if frontier.isEmpty():
            return []

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    from game import Directions
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self,initial_state,parent=None,action=None):
            self.state=initial_state
            self.parent=parent
            self.action=action
    node = Node(problem.getStartState())
    if problem.isGoalState(node.state):
        return solution(node)
    frontier = Queue()
    frontier.push(node)
    explored=[]
    while True:
        node = frontier.pop()
        if node.state not in explored:
            if problem.isGoalState(node.state):
                return solution(node)
            explored.append(node.state)
            for child in problem.getSuccessors(node.state):
                childnode = Node(child[0], node, child[1])
                if (childnode.state not in explored) and (childnode not in frontier.list):
                    frontier.push(childnode)
        if frontier.isEmpty():
            return []
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    from game import Directions
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self,initial_state,cost,parent=None,action=None):
            self.state=initial_state
            self.parent=parent
            self.action=action
            self.cost=cost
    node = Node(problem.getStartState(),0)
    if problem.isGoalState(node.state):
        return solution(node)
    frontier = PriorityQueue()
    frontier.push(node,0)
    explored=set()
    while True:
        node = frontier.pop()
        if node.state not in explored:
            if problem.isGoalState(node.state):
                return solution(node)
            explored.add(node.state)
            for child in problem.getSuccessors(node.state):
                childnode = Node(child[0], node.cost + child[2], node, child[1])
                if (childnode.state not in explored) and (childnode not in frontier.heap):
                    frontier.update(childnode, childnode.cost)
        if frontier.isEmpty():
            return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    from game import Directions
    "*** YOUR CODE HERE ***"
    class Node:
        def __init__(self,initial_state,cost,heuristic,parent=None,action=None):
            self.state=initial_state
            self.parent=parent
            self.action=action
            self.cost=cost
            self.heuristic=heuristic

    node = Node(problem.getStartState(), 0,heuristic(problem.getStartState(),problem))
    if problem.isGoalState(node.state):
        return solution(node)
    frontier = PriorityQueue()
    frontier.push(node,node.heuristic)
    explored = {}
    frontierset={}
    while True:
        node = frontier.pop()
        if node.state not in explored:
            if problem.isGoalState(node.state):
                return solution(node)
            explored[node.state]=1
            #explored.append(node)
            for child in problem.getSuccessors(node.state):
                childnode = Node(child[0], node.cost + child[2], heuristic(child[0], problem), node, child[1])
                if (childnode.state not in explored) and (childnode not in frontierset):
                    #frontier.update(childnode, childnode.cost + childnode.heuristic-len(frontier.heap)*0.00001)
                    frontier.update(childnode, childnode.cost + childnode.heuristic)
                    frontierset[childnode]=1
        if frontier.isEmpty():
            return []

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


def solution(node):
    route = []
    while node.parent != None:
        route.append(node.action)
        node = node.parent
    route = route[::-1]
    return route