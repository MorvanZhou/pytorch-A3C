import argparse
import os

import torch.multiprocessing as mp

from continuous_A3C import a3c

NAME_PENDULUM = "Pendulum-v0"
NAME_HUMANOID = "Humanoid-v3"
NAME_STANDUP = "HumanoidStandup-v3"
NAME_ANT = "Ant-v3"
ADV_SIMPLE = "simple"
ADV_GAE = "gae"

if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "1"

    # Parse the command line arguments
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--name", type=str, help="Name of environment", default=NAME_PENDULUM,
                         choices=[NAME_PENDULUM, NAME_HUMANOID, NAME_STANDUP, NAME_ANT])
    _parser.add_argument("--advantage", type=str, help="Type of advantage estimator", default="gae",
                         choices=[ADV_SIMPLE, ADV_GAE])
    _parser.add_argument("--episodes", type=int, help="Number of episodes", default=1000)
    _parser.add_argument("--workers", type=int, help="Number of worker threads", default=mp.cpu_count())
    _parser.add_argument("--gam", type=float, help="Gamma", default=0.9)
    _parser.add_argument("--lam", type=float, help="Lambda", default=0.99)
    _args = _parser.parse_args()

    a3c(_args.workers, _args.gam, _args.lam, _args.episodes, _args.name, _args.advantage == ADV_GAE)
