import os
import pickle
import tkinter as tk
from tkinter.filedialog import askopenfilenames

import matplotlib.pyplot as plt

from main import ADV_GAE
from utils import RESULTS_FOLDER


def compare(filenames: list) -> None:
    env = ""
    for filename in filenames:
        env, eps, workers, adv, gam, lam = split_name(filename)
        with open(filename, "rb") as f_in:
            unpickled = pickle.load(f_in)
        plt.plot(unpickled)
        plt.plot(unpickled, label=f"{workers}_{adv}_{gam}{f'_{lam}' if adv == ADV_GAE else ''}")

    plt.title(env)
    plt.legend()
    plt.show()


def split_name(name: str):
    """
    ['Pendulum', '1000', '4', 'Simple', '0.9']
    """
    data = os.path.basename(name).split("_")
    data[-1] = data[-1][:data[-1].index(".pkl")]
    if len(data) == 5:
        data.append("")
    return tuple(data)


if __name__ == "__main__":
    _root = tk.Tk()
    _root.withdraw()
    _root.filenames = askopenfilenames(initialdir=RESULTS_FOLDER,
                                       title="Select files",
                                       filetypes=[("Pickle files", "*.pkl")])
    _filenames = [s for s in _root.filenames]
    _root.destroy()
    compare(_filenames)
