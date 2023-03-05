import os
import pickle
import tkinter as tk
from tkinter.filedialog import askopenfilenames

import matplotlib.pyplot as plt

from utils import RESULTS_FOLDER


def normalise(results: list):
    u_bound = max(results)
    l_bound = min(results)
    delta = u_bound - l_bound
    return [(v - l_bound) / delta for v in results]


def plot(title: str, data, norm=False) -> None:
    for item in data:
        with open(item[0], "rb") as f_in:
            unpickled = pickle.load(f_in)
        plt.plot(normalise(unpickled) if norm else unpickled, label=item[1])

    plt.title(title)
    plt.legend()
    plt.show()


def create_label(filename: str, show_adv: bool, show_gam: bool, show_lam: bool, show_wrk: bool):
    _, eps, workers, adv, gam, lam = split_name(filename)
    label = ""
    if show_adv:
        label += adv
    if show_gam:
        if len(label) > 0:
            label += ", "
        label += f"γ={gam}"
    if show_lam:
        if len(label) > 0:
            label += ", "
        label += f"λ={lam}"
    if show_wrk:
        if len(label) > 0:
            label += ", "
        label += f"W={workers}"
    return label


def split_name(name: str):
    """
    ['Pendulum', '1000', '4', 'Simple', '0.9']
    """
    data = os.path.basename(name).split("_")
    data[-1] = data[-1][:data[-1].index(".pkl")]
    if len(data) == 5:
        data.append("")
    return tuple(data)


def choose_files(title, adv, gam, lam, wrk, nrm):
    filenames = [f for f in askopenfilenames(initialdir=RESULTS_FOLDER,
                                             title="Select files",
                                             filetypes=[("Pickle files", "*.pkl")])]
    if len(filenames) > 0:
        labels = [create_label(f, adv, gam, lam, wrk) for f in filenames]
        plot(title, zip(filenames, labels), nrm)


def init_gui():
    root = tk.Tk()
    current_row = 0

    # Title
    label_title = tk.Label(root, text="Title:")
    label_title.grid(row=current_row, column=0)

    entry_title = tk.Entry(root)
    entry_title.grid(row=current_row, column=1)

    current_row += 1

    # Advantage
    var_adv = tk.IntVar()
    checkbox_adv = tk.Checkbutton(root, text="Advantage", variable=var_adv)
    checkbox_adv.grid(row=current_row, column=1)

    current_row += 1

    # Gamma
    var_gam = tk.IntVar()
    checkbox_gam = tk.Checkbutton(root, text="Gamma", variable=var_gam)
    checkbox_gam.grid(row=current_row, column=1)

    current_row += 1

    # Lambda
    var_lam = tk.IntVar()
    checkbox_lam = tk.Checkbutton(root, text="Lambda", variable=var_lam)
    checkbox_lam.grid(row=current_row, column=1)

    current_row += 1

    # Workers
    var_wrk = tk.IntVar()
    checkbox_wrk = tk.Checkbutton(root, text="Workers", variable=var_wrk)
    checkbox_wrk.grid(row=current_row, column=1)

    current_row += 1

    # Normalise
    var_nrm = tk.IntVar()
    var_nrm.set(1)
    checkbox_nrm = tk.Checkbutton(root, text="Normalise?", variable=var_nrm)
    checkbox_nrm.grid(row=current_row, column=1)

    current_row += 1

    # File picker
    button_choose_files = tk.Button(root, text="Choose files", command=lambda: choose_files(entry_title.get(),
                                                                                            var_adv.get() == 1,
                                                                                            var_gam.get() == 1,
                                                                                            var_lam.get() == 1,
                                                                                            var_wrk.get() == 1,
                                                                                            var_nrm.get() == 1))
    button_choose_files.grid(row=current_row, column=0)

    # Quit
    button_quit = tk.Button(root, text="Exit", command=root.destroy)
    button_quit.grid(row=current_row, column=2)

    return root


if __name__ == "__main__":
    _root = init_gui()
    _root.mainloop()
