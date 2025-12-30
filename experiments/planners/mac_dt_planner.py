from experiments.planners.base import BasePlanner

class MACDTPlanner(BasePlanner):
    def reset(self, env, starts):
        # init GP, counters, episode state
        pass

    def step(self, t, env):
        # 1. if new episode → oracle
        # 2. move agents one step
        # 3. sample max-uncertainty
        # 4. check doubling condition
        pass