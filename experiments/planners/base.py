class BasePlanner:
    def reset(self, env, args):
        raise NotImplementedError

    def step(self, t):
        raise NotImplementedError

    def finalize(self):
        return {}