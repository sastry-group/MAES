from .ergodic_planner import ErgodicPlanner
from .mac_dt_planner import MACDTPlanner

def make_planner(args):
    if args.planner == "ergodic":
        return ErgodicPlanner(args)
    elif args.planner == "mac_dt":
        return MACDTPlanner(args)
    else:
        raise ValueError(f"Unknown planner {args.planner}")