import search
import random
import math


class PlanningProblem(search.Problem):
    """This class implements a Planning problem"""
    def __init__(self, initial, goal, skills):
        """Don't forget to set the goal or implement the goal test
        You should change the initial to your own representation"""
        initial = (initial[0], tuple(initial[1].items()))
        search.Problem.__init__(self, initial, None)

        self.skills = skills
        self.goal = goal
        self.success = self.goal_test(initial)
        
    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a tuple, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        try:
            actions = list(self.skills[state[0]])
        except:
            actions = []
        new_actions = []
        for action in actions:
            action = (action[0], tuple(action[1].items()), action[2])
            new_actions.append(action)

        return new_actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        state = list(state)

        state[1] = dict(state[1])
        # action[1] = dict(action[1])

        state[0] = action[0]
        state[1].update(action[1])

        state[1] = tuple(state[1].items())
        state = tuple(state)

        return state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state, compares to the created goal state"""
        state = list(state)
        state[1] = dict(state[1])
        test1 = False
        test2 = False

        if state[0] == self.goal[0]:
            test1 = True
        if state[1] == self.goal[1]:
            test2 = True

        return test1 and test2

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        h = 0
        "goal heuristic"
        if self.goal_test(node.state):
            return h
        else:
            return 1

        # TODO-> make heuristic based on number of world goals left times the total number of skills
        pass

    """Feel free to add your own functions"""


def create_planning_problem(problem, goal, skill_dict):
 
    return PlanningProblem(problem, goal, skill_dict)

