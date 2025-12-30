from .ergodic_planner import ErgodicPlanner
from .mac_dt_planner import MACDTPlanner

def make_planner(args):
    if args.method in ["mh", "remc"]:
        return ErgodicPlanner(args)
    elif args.method == "mac_dt":
        return MACDTPlanner(args)
    else:
        raise ValueError(f"Unknown method {args.method}")

def make_planner(args):
    if args.method in ["mh", "remc"]:
        return ErgodicPlanner(args)
    elif args.method == "mac_dt":
        return MACDTPlanner(args)