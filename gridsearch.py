import argparse

import torch.multiprocessing as mp

from continuous_A3C import a3c
from main import ADV_SIMPLE, ADV_GAE, NAME_PENDULUM, NAME_HUMANOID, NAME_STANDUP, NAME_ANT

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--name", type=str, help="Name of environment", default=NAME_PENDULUM,
                         choices=[NAME_PENDULUM, NAME_HUMANOID, NAME_STANDUP, NAME_ANT])
    _parser.add_argument("--episodes", type=int, help="Number of episodes", default=1000)
    _parser.add_argument("-ws", "--workers", nargs="+", help="Gamma values", required=True,
                         choices=[str(w) for w in range(1, mp.cpu_count() + 1)])
    _parser.add_argument("-as", "--advantages", nargs="+", help="Advantages", required=True,
                         choices=[ADV_SIMPLE, ADV_GAE])
    _parser.add_argument("-gs", "--gammas", nargs="+", help="Gamma values", required=True)
    _parser.add_argument("-ls", "--lambdas", nargs="+", help="Lambda values (only for GAE)", required=False)
    _args = _parser.parse_args()

    _env_name = _args.name
    _num_episodes = _args.episodes
    _workers = [int(w) for w in _args.workers]
    _advantages = [s.lower() for s in _args.advantages]
    _gammas = [float(g) for g in _args.gammas]
    _lambdas = [float(l) for l in _args.lambdas] if _args.lambdas else None

    for _w in _workers:
        for _a in _advantages:
            for _g in _gammas:
                if _a == ADV_GAE:
                    for _l in _lambdas:
                        print(_w, _a, _g, _l)
                        a3c(_w, _g, _l, _num_episodes, _env_name, True, False)
                else:
                    print(_w, _a, _g)
                    a3c(_w, _g, 0.0, _num_episodes, _env_name, False, False)
