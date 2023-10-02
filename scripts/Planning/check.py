import planner
import search
import time
#import networks
from graph_components import *


def timeout_exec(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.result = (-3, -3, e)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return default
    else:
        return it.result


def check_problem(p, search_method, timeout):
    """ Constructs a problem using ex1.create_poisonserver_problem,
    and solves it using the given search_method with the given timeout.
    Returns a tuple of (solution length, solution time, solution)
    (-2, -2, None) means there was a timeout
    (-3, -3, ERR) means there was some error ERR during search"""

    t1 = time.time()
    s = timeout_exec(search_method, args=[p], timeout_duration=timeout)
    t2 = time.time()

    if isinstance(s, search.Node):
        solve = s
        solution = list(map(lambda n: n.action, solve.path()))[1:]
        return (len(solution), t2 - t1, solution)
    elif s is None:
        return (-2, -2, None)
    else:
        return s


def solve_problems(initial, goal, skills):
    solved = 0

    try:
        p = planner.create_planning_problem(initial, goal, skills)
    except Exception as e:
        print("Error creating problem: ", e)
        return None
    timeout = 60
    result = check_problem(p, search.astar_search, timeout)

    return result[2]


def main():
    robot_initial = s_11
    world_initial = {"cube": "counter"}
    initial = (robot_initial, world_initial)
    skills = skill_dict
    robot_final_state = s_16
    world_final_state = {"cube": "pickup"}
    goal = (robot_final_state, world_final_state)

    solve_problems(initial, goal, skills)


if __name__ == '__main__':
    main()
